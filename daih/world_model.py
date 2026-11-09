"""
world_model.py

Neuro-symbolic WorldModel with:
- variable-size matrix inputs;
- state and layer identifiers;
- four types: f, g, p, and V;
- optional local functions, indexed by (state_id, layer_id, type);
- global neural prediction when no local function is registered.

Local functions are user-defined Python callables.
They are neither trained nor registered in state_dict().
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn

from .policy_generator import PolicyGenerator, PolicyOutput


FunctionType = str
LocalFunction = Callable[..., Union[torch.Tensor, PolicyOutput, tuple, list]]


@dataclass
class WorldModelOutput:
    function_type: str
    values: Optional[Any] = None
    policy: Optional[PolicyOutput] = None
    used_local_function: bool = False
    state_id: Optional[int] = None
    layer_id: Optional[int] = None


class WorldModel(nn.Module):
    """
    Global prediction model and registry for local mathematical functions.

    Main API:
        model(state, state_id, layer_id, function_type, ...)

    Matrix inputs can have any spatial size
    (2D [R,C] or 3D batch [B,R,C]). Within the same batch, matrices must
    naturally have the same shape.
    """
    VALID_TYPES = ("f", "g", "p", "V")

    def __init__(
        self,
        hidden_dim: int = 256,
        id_embedding_dim: int = 24,
        id_buckets: int = 4096,
        policy_hidden_dim: int = 256,
        vocab_size: int = 32,
    ):
        super().__init__()
        if min(hidden_dim, id_embedding_dim, id_buckets, policy_hidden_dim, vocab_size) <= 0:
            raise ValueError("All dimensions must be positive.")

        self.hidden_dim = hidden_dim
        self.id_buckets = id_buckets
        self.state_embedding = nn.Embedding(id_buckets, id_embedding_dim)
        self.layer_embedding = nn.Embedding(id_buckets, id_embedding_dim)
        self.type_embedding = nn.Embedding(len(self.VALID_TYPES), id_embedding_dim)

        # Input statistics independent of the number of elements:
        # mean, standard deviation, min, max, log(number of rows), log(number of columns).
        feature_dim = 6 + 3 * id_embedding_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Decoder f: one output per element of the input matrix.
        self.f_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2 or 1),
            nn.GELU(),
            nn.Linear(hidden_dim // 2 or 1, 1),
        )

        # Decoder g/V: one output per coordinate of the requested shape.
        self.gv_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.local_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 7, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.policy_generator = PolicyGenerator(
            input_size=hidden_dim,
            hidden_dim=policy_hidden_dim,
            vocab_size=vocab_size,
        )

        # Local callables are intentionally separated from PyTorch parameters:
        # they represent the provided rules/mathematics.
        self._local_functions: Dict[Tuple[int, int, str], LocalFunction] = {}

    @staticmethod
    def _type_index(function_type: str) -> int:
        try:
            return WorldModel.VALID_TYPES.index(function_type)
        except ValueError as exc:
            raise ValueError(
                f"Unknown type {function_type!r}. Accepted types: "
                f"{WorldModel.VALID_TYPES}."
            ) from exc

    def register_local_function(
        self,
        state_id: int,
        layer_id: int,
        function_type: str,
        function: LocalFunction,
        *,
        replace: bool = True,
    ) -> None:
        """Registers a local function for a state, layer, and type."""
        self._type_index(function_type)
        if not callable(function):
            raise TypeError("function must be callable.")
        key = (int(state_id), int(layer_id), function_type)
        if key in self._local_functions and not replace:
            raise KeyError(f"A local function already exists for {key}.")
        self._local_functions[key] = function

    def unregister_local_function(
        self, state_id: int, layer_id: int, function_type: str
    ) -> None:
        key = (int(state_id), int(layer_id), function_type)
        self._local_functions.pop(key, None)

    def has_local_function(self, state_id: int, layer_id: int, function_type: str) -> bool:
        return (int(state_id), int(layer_id), function_type) in self._local_functions

    def _as_batch(self, state: torch.Tensor):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)

        if not state.is_floating_point():
            state = state.float()

        original_dim = state.dim()

        if original_dim == 1:
            # [C] -> [1, 1, C]
            state = state.unsqueeze(0).unsqueeze(0)

        elif original_dim == 2:
            # [R, C] -> [1, R, C]
            state = state.unsqueeze(0)

        elif original_dim != 3:
            raise ValueError(
                "state must be a vector [C], a matrix [R,C], "
                "or a batch [B,R,C]."
            )

        if state.shape[1] < 1 or state.shape[2] < 1:
            raise ValueError("The input matrix cannot be empty.")

        return state, original_dim
        

    def _id_tensor(self, value, batch_size, device):
        ids = torch.as_tensor(value, device=device, dtype=torch.long)
        if ids.ndim == 0:
            ids = ids.expand(batch_size)
        if ids.shape != (batch_size,):
            raise ValueError("state_id and layer_id must be scalars or of shape [B].")
        # Deterministic hashing: allows accepting new IDs without
        # resizing the embeddings. Collisions remain possible.
        return torch.remainder(ids, self.id_buckets)

    def encode_context(
        self, state: torch.Tensor, state_id, layer_id, function_type: str
    ) -> torch.Tensor:
        """Encodes the matrix and its metadata into a context [B, hidden_dim]."""
        self._type_index(function_type)
        x, _ = self._as_batch(state)
        batch = x.shape[0]
        state_ids = self._id_tensor(state_id, batch, x.device)
        layer_ids = self._id_tensor(layer_id, batch, x.device)
        type_ids = torch.full(
            (batch,), self._type_index(function_type), device=x.device, dtype=torch.long
        )
        flat = x.reshape(batch, -1)
        stats = torch.stack(
            [
                flat.mean(dim=1),
                flat.std(dim=1, unbiased=False),
                flat.amin(dim=1),
                flat.amax(dim=1),
                torch.full_like(flat[:, 0], float(x.shape[1])).log1p(),
                torch.full_like(flat[:, 0], float(x.shape[2])).log1p(),
            ],
            dim=1,
        )
        features = torch.cat(
            [
                stats,
                self.state_embedding(state_ids),
                self.layer_embedding(layer_ids),
                self.type_embedding(type_ids),
            ],
            dim=1,
        )
        return self.context_encoder(features)

    @staticmethod
    def _grid(batch, rows, cols, device, dtype):
        r = torch.linspace(-1.0, 1.0, rows, device=device, dtype=dtype)
        c = torch.linspace(-1.0, 1.0, cols, device=device, dtype=dtype)
        rr, cc = torch.meshgrid(r, c, indexing="ij")
        return torch.stack(
            [
                rr,
                cc,
                torch.full_like(rr, rows / max(rows, cols)),
                torch.full_like(cc, cols / max(rows, cols)),
            ],
            dim=-1,
        ).reshape(1, rows * cols, 4).expand(batch, -1, -1)

    def _predict_f(
        self, x: torch.Tensor, context: torch.Tensor,
        local_values: Optional[torch.Tensor] = None,
        local_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, rows, cols = x.shape
        r = torch.linspace(-1.0, 1.0, rows, device=x.device, dtype=x.dtype)
        c = torch.linspace(-1.0, 1.0, cols, device=x.device, dtype=x.dtype)
        rr, cc = torch.meshgrid(r, c, indexing="ij")
        coords = torch.stack([rr, cc], dim=-1).reshape(1, rows * cols, 2)
        coords = coords.expand(batch, -1, -1)
        values = x.reshape(batch, rows * cols, 1)
        ctx = context[:, None, :].expand(-1, rows * cols, -1)
        if local_values is None:
            local_values = torch.zeros_like(x)
        if local_mask is None:
            local_mask = torch.zeros_like(x)
        local_values = local_values.reshape(batch, rows * cols, 1)
        local_mask = local_mask.reshape(batch, rows * cols, 1)
        pred = self.f_decoder(torch.cat(
            [ctx, values, coords, local_values, local_mask], dim=-1
        ))
        return pred.reshape(batch, rows, cols)

    def _predict_gv(
        self, context: torch.Tensor, output_shape: Tuple[int, int],
        local_values: Optional[torch.Tensor] = None,
        local_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rows, cols = map(int, output_shape)
        if rows <= 0 or cols <= 0:
            raise ValueError("output_shape must contain two positive dimensions.")
        batch = context.shape[0]
        coords = self._grid(batch, rows, cols, context.device, context.dtype)
        ctx = context[:, None, :].expand(-1, rows * cols, -1)
        if local_values is None:
            local_values = torch.zeros(batch, rows, cols, device=context.device, dtype=context.dtype)
        if local_mask is None:
            local_mask = torch.zeros_like(local_values)
        local_features = torch.stack(
            [local_values.reshape(batch, -1), local_mask.reshape(batch, -1)], dim=-1
        )
        pred = self.gv_decoder(torch.cat([ctx, coords, local_features], dim=-1))
        return pred.reshape(batch, rows, cols)

    @staticmethod
    def _restore_batch(tensor: torch.Tensor, original_dim: int):
        if original_dim == 1 and tensor.shape[0] == 1:
            return tensor.squeeze(0).squeeze(0)

        if original_dim == 2 and tensor.shape[0] == 1:
            return tensor.squeeze(0)

        return tensor

    def forward(
        self,
        state: torch.Tensor,
        state_id: int,
        layer_id: int,
        function_type: str,
        *,
        action: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        output_shape: Optional[Tuple[int, int]] = None,
        output_shapes: Optional[Tuple[Tuple[int, int], ...]] = None,
        max_steps: int = 20,
        deterministic: Optional[bool] = None,
    ) -> WorldModelOutput:
        """
        Executes local functions and completes masked values with neural predictions.

        For f/V and single-output g, a local function may return values or
        (values, mask). For multi-output g, it returns one such item per
        declared output, in output_shapes order. Masks are consumed internally:
        this method exposes only the final value matrix/matrices. mask=1 keeps
        the local value; mask=0 requests a neural prediction. Type p retains
        its policy interface.
        """
        self._type_index(function_type)
        x, original_2d = self._as_batch(state)
        local = self._local_functions.get((int(state_id), int(layer_id), function_type))

        if local is not None:
            local_input = self._restore_batch(x, original_2d)

            if function_type == "f" and action is not None:
                local_result = local(local_input, action=action)

            elif function_type == "p":
                local_result = local(
                    local_input,
                    temperature=temperature
                )

            else:
                local_result = local(local_input)

            
            if function_type == "p":

                if isinstance(local_result, PolicyOutput):

                    # The generator has already built its PolicyOutput.
                    policy_result = local_result

                elif isinstance(local_result, tuple) and len(local_result) == 2:

                    # Return: (policy matrix, global score G)
                    matrix, scalar = local_result

                    matrix = torch.as_tensor(
                        matrix, device=x.device, dtype=x.dtype
                    )

                    if matrix.ndim == 2:
                        matrix = matrix.unsqueeze(0)

                    if matrix.ndim != 3:
                        raise ValueError(
                            "A local policy must be of shape [L,V] or [B,L,V]."
                        )

                    scalar = torch.as_tensor(
                        scalar, device=x.device, dtype=x.dtype
                    )

                    if scalar.ndim == 0:
                        scalar = scalar.reshape(1, 1)
                    elif scalar.ndim == 1:
                        scalar = scalar.unsqueeze(-1)

                    # Detection of actually active steps.
                    temporal_mask = (
                        matrix.abs().sum(dim=-1) > 0
                    ).to(x.dtype)

                    policy_result = PolicyOutput(
                        policy=matrix,
                        temporal_mask=temporal_mask,
                        G=scalar,
                    )

                else:

                    # Return: policy matrix alone.
                    matrix = torch.as_tensor(
                        local_result, device=x.device, dtype=x.dtype
                    )

                    if matrix.ndim == 2:
                        matrix = matrix.unsqueeze(0)

                    if matrix.ndim != 3:
                        raise ValueError(
                            "A local policy must be of shape [L,V] or [B,L,V]."
                        )

                    temporal_mask = (
                        matrix.abs().sum(dim=-1) > 0
                    ).to(x.dtype)

                    # A matrix alone does not contain neural scores.
                    # We therefore initialize G to zero rather than inventing a score.
                    G = torch.zeros(
                        (matrix.shape[0], 1),
                        device=x.device,
                        dtype=x.dtype,
                    )

                    policy_result = PolicyOutput(
                        policy=matrix,
                        temporal_mask=temporal_mask,
                        G=G,
                    )

                return WorldModelOutput(
                    function_type="p",
                    policy=policy_result,
                    used_local_function=True,
                    state_id=int(state_id),
                    layer_id=int(layer_id),
                )


            # g may declare several independently-shaped outputs. Each item
            # is either a matrix or a (matrix, binary_mask) pair.
            if function_type == "g" and output_shapes is not None:
                shapes = tuple(tuple(map(int, shape)) for shape in output_shapes)
                if not shapes:
                    raise ValueError("output_shapes cannot be empty for g.")

                if len(shapes) == 1:
                    # Backward compatibility: one output may be returned as
                    # a matrix, or as (matrix, mask).
                    output_items = [local_result]
                    if isinstance(local_result, (tuple, list)) and len(local_result) == 1:
                        output_items = [local_result[0]]
                elif isinstance(local_result, (tuple, list)) and len(local_result) == len(shapes):
                    output_items = list(local_result)
                elif isinstance(local_result, torch.Tensor):
                    # Legacy multi-output tensor: split its columns when all
                    # outputs share the same row count.
                    if len({shape[0] for shape in shapes}) != 1:
                        raise ValueError(
                            "A single tensor cannot represent g outputs with "
                            "different row counts; return one matrix per output."
                        )
                    expected_cols = sum(shape[1] for shape in shapes)
                    if local_result.shape[-1] != expected_cols:
                        raise ValueError(
                            f"Legacy g tensor has {local_result.shape[-1]} columns; "
                            f"{expected_cols} are required by output_shapes."
                        )
                    output_items = list(torch.split(
                        local_result, [shape[1] for shape in shapes], dim=-1
                    ))
                else:
                    raise ValueError(
                        f"g declares {len(shapes)} outputs; the local function "
                        "must return one item per output."
                    )

                merged_outputs = []
                for index, (item, expected_shape) in enumerate(zip(output_items, shapes)):
                    if isinstance(item, tuple) and len(item) == 2:
                        local_values, local_mask = item
                    else:
                        local_values, local_mask = item, None

                    values = torch.as_tensor(local_values, device=x.device, dtype=x.dtype)
                    if values.dim() == 2:
                        values = values.unsqueeze(0)
                    if values.dim() != 3 or tuple(values.shape[-2:]) != expected_shape:
                        raise ValueError(
                            f"Local g output #{index} must have shape [R,C] or [B,R,C] "
                            f"with spatial shape {expected_shape}; received {tuple(values.shape)}."
                        )
                    if values.shape[0] not in (1, x.shape[0]):
                        raise ValueError(
                            "The batch size of each local g output must be 1 "
                            "or equal to that of state."
                        )
                    if values.shape[0] == 1 and x.shape[0] > 1:
                        values = values.expand(x.shape[0], -1, -1)

                    if local_mask is None:
                        mask = torch.ones_like(values)
                    else:
                        mask = torch.as_tensor(local_mask, device=x.device, dtype=x.dtype)
                        if mask.dim() == 2:
                            mask = mask.unsqueeze(0)
                        try:
                            mask = torch.broadcast_to(mask, values.shape)
                        except RuntimeError as exc:
                            raise ValueError(
                                f"Mask for g output #{index} must be broadcastable "
                                f"to {tuple(values.shape)}; received {tuple(mask.shape)}."
                            ) from exc
                        if not bool(torch.all((mask == 0) | (mask == 1))):
                            raise ValueError("Local masks must be binary (0 or 1).")

                    if bool(torch.all(mask == 1)):
                        merged = values
                    else:
                        context = self.encode_context(x, state_id, layer_id, function_type)
                        flat_local = values.reshape(values.shape[0], -1)
                        local_stats = torch.stack(
                            [
                                flat_local.mean(dim=1),
                                flat_local.std(dim=1, unbiased=False),
                                flat_local.amin(dim=1),
                                flat_local.amax(dim=1),
                                torch.full_like(flat_local[:, 0], float(values.shape[1])).log1p(),
                                torch.full_like(flat_local[:, 0], float(values.shape[2])).log1p(),
                                mask.reshape(mask.shape[0], -1).mean(dim=1),
                            ],
                            dim=1,
                        )
                        fused_context = self.local_fusion(
                            torch.cat([context, local_stats], dim=1)
                        )
                        predicted = self._predict_gv(
                            fused_context, expected_shape, values, mask
                        )
                        merged = mask * values + (1.0 - mask) * predicted

                    # The mask is an internal mechanism: only the merged
                    # matrix is exposed by WorldModel.
                    merged = self._restore_batch(merged, original_2d)
                    merged_outputs.append(merged)

                output_values = (
                    merged_outputs[0] if len(merged_outputs) == 1
                    else tuple(merged_outputs)
                )
                return WorldModelOutput(
                    function_type="g", values=output_values, used_local_function=True,
                    state_id=int(state_id), layer_id=int(layer_id)
                )

            # Original single-output path used by f and V, and by legacy g
            # calls that do not provide output_shapes.
            if isinstance(local_result, tuple) and len(local_result) == 2:
                local_values, local_mask = local_result
            else:
                local_values, local_mask = local_result, None

            values = torch.as_tensor(local_values, device=x.device, dtype=x.dtype)
            expected_shape = tuple(x.shape[1:]) if function_type == "f" else output_shape
            if expected_shape is None:
                raise ValueError("For a local function g or V, provide output_shape.")
            if values.dim() == 2:
                values = values.unsqueeze(0)
            if values.dim() != 3 or tuple(values.shape[-2:]) != tuple(expected_shape):
                raise ValueError(
                    f"The local function {function_type} must return [R,C] or [B,R,C] "
                    f"with a spatial shape {tuple(expected_shape)}; received {tuple(values.shape)}."
                )
            if values.shape[0] not in (1, x.shape[0]):
                raise ValueError("The batch size of the local output must be equal to 1 or to that of state.")
            if values.shape[0] == 1 and x.shape[0] > 1:
                values = values.expand(x.shape[0], -1, -1)

            if local_mask is None:
                mask = torch.ones_like(values)
            else:
                mask = torch.as_tensor(local_mask, device=x.device, dtype=x.dtype)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                try:
                    mask = torch.broadcast_to(mask, values.shape)
                except RuntimeError as exc:
                    raise ValueError(
                        f"The mask must be broadcastable to {tuple(values.shape)}; "
                        f"received shape: {tuple(mask.shape)}."
                    ) from exc
                if not bool(torch.all((mask == 0) | (mask == 1))):
                    raise ValueError("The local mask must be binary (0 or 1).")

            if bool(torch.all(mask == 1)):
                merged = values
            else:
                context = self.encode_context(x, state_id, layer_id, function_type)
                flat_local = values.reshape(values.shape[0], -1)
                local_stats = torch.stack(
                    [
                        flat_local.mean(dim=1),
                        flat_local.std(dim=1, unbiased=False),
                        flat_local.amin(dim=1),
                        flat_local.amax(dim=1),
                        torch.full_like(flat_local[:, 0], float(values.shape[1])).log1p(),
                        torch.full_like(flat_local[:, 0], float(values.shape[2])).log1p(),
                        mask.reshape(mask.shape[0], -1).mean(dim=1),
                    ],
                    dim=1,
                )
                context = self.local_fusion(torch.cat([context, local_stats], dim=1))
                predicted = (
                    self._predict_f(x, context, values, mask)
                    if function_type == "f"
                    else self._predict_gv(context, expected_shape, values, mask)
                )
                merged = mask * values + (1.0 - mask) * predicted

            # Masks are consumed internally and are never part of the
            # WorldModel public output.
            merged = self._restore_batch(merged, original_2d)
            return WorldModelOutput(
                function_type=function_type, values=merged, used_local_function=True,
                state_id=int(state_id), layer_id=int(layer_id)
            )

        # No local callable: global computation takes over for the entire output.
        context = self.encode_context(x, state_id, layer_id, function_type)

        if function_type == "p":
            policy = self.policy_generator(
                context, max_steps=max_steps, deterministic=deterministic, temperature=temperature
            )
            return WorldModelOutput(
                function_type="p", policy=policy, used_local_function=False,
                state_id=int(state_id), layer_id=int(layer_id)
            )

        if function_type == "f":
            values = self._predict_f(x, context)
        elif function_type == "g" and output_shapes is not None:
            shapes = tuple(tuple(map(int, shape)) for shape in output_shapes)
            if not shapes:
                raise ValueError("output_shapes cannot be empty for g.")
            predictions = [
                self._restore_batch(self._predict_gv(context, shape), original_2d)
                for shape in shapes
            ]
            values = predictions[0] if len(predictions) == 1 else tuple(predictions)
        else:
            if output_shape is None:
                raise ValueError(
                    f"output_shape=(rows, cols) is mandatory for {function_type} "
                    "when no local function is defined."
                )
            values = self._predict_gv(context, output_shape)

        if not (function_type == "g" and output_shapes is not None):
            values = self._restore_batch(values, original_2d)
        return WorldModelOutput(
            function_type=function_type, values=values, used_local_function=False,
            state_id=int(state_id), layer_id=int(layer_id)
        )