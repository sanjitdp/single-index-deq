import torch


def newton_solver(func, initial, max_iter=30, tol=1e-6, **kwargs):
    state = initial.detach().clone()
    shape = state.shape
    batch = len(state)
    eye = torch.eye(40, device=state.device, dtype=state.dtype)[None]
    for outer in range(min(max_iter, 30)):
        value = func(state)
        residual = (value - state).flatten(1)
        norm = residual.norm(dim=1)
        if (norm / (value.flatten(1).norm(dim=1) + 1e-9)).max().item() <= tol:
            break

        def operator(v):

            _, jv = torch.autograd.functional.jvp(
                func, state, v.reshape(shape), create_graph=False, strict=False
            )
            return v - jv.flatten(1)

        basis = [residual / norm.clamp_min(1e-20)[:, None]]
        hessenberg = state.new_zeros(batch, 41, 40)
        rhs = state.new_zeros(batch, 41, 1)
        rhs[:, 0, 0] = norm
        coefficients = None
        for column in range(40):
            vector = operator(basis[column])

            for _ in range(2):
                for row in range(column + 1):
                    projection = (basis[row] * vector).sum(dim=1)
                    hessenberg[:, row, column] += projection
                    vector = vector - projection[:, None] * basis[row]
            length = vector.norm(dim=1)
            hessenberg[:, column + 1, column] = length
            basis.append(vector / length.clamp_min(1e-20)[:, None])
            matrix = hessenberg[:, : column + 2, : column + 1]
            target = rhs[:, : column + 2]
            gram = matrix.transpose(1, 2) @ matrix
            scale = gram.diagonal(dim1=1, dim2=2).amax(dim=1).clamp_min(1e-12)

            coefficients = torch.linalg.solve(
                gram + 1e-6 * scale[:, None, None] * eye[:, : column + 1, : column + 1],
                matrix.transpose(1, 2) @ target,
            )
            linear_relative = (matrix @ coefficients - target).flatten(1).norm(
                dim=1
            ) / norm.clamp_min(1e-20)
            if linear_relative.max().item() <= 0.1:
                break
        direction = sum(
            coefficients[:, j, 0, None] * basis[j] for j in range(column + 1)
        )
        best = state
        best_norm = norm
        for backtrack in range(9):
            candidate = state + (2.0 ** (-backtrack)) * direction.reshape(shape)
            candidate_value = func(candidate)
            candidate_norm = (candidate_value - candidate).flatten(1).norm(dim=1)
            improved = torch.isfinite(candidate_norm) & (candidate_norm < best_norm)
            best = torch.where(
                improved.reshape(-1, *([1] * (state.ndim - 1))), candidate, best
            )
            best_norm = torch.where(improved, candidate_norm, best_norm)
            if (best_norm <= 0.9 * norm).all():
                break
        state = best.detach()
    return state, [], {}
