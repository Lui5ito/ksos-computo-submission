from scipy.optimize import minimize
from types import SimpleNamespace
import numpy as np

def create_clip_bounds(bounds_list, default_inf):
    bounds_array = np.array(bounds_list, dtype=object)
    is_none = np.equal(bounds_array, None)
    final_bounds = np.where(is_none, default_inf, bounds_array).astype(float)
    return final_bounds


def solve_dispatch(solver_name, **kwargs):
        if solver_name == "BFGS":
            return solve_bfgs(**kwargs)
        elif solver_name == "AGD":
            return solve_agd(**kwargs)
        else:
            raise ValueError(f"Solver {solver_name} not implemented")


def solve_bfgs(initial_value, bounds, obj_func, obj_grad, gap_func, max_iters, solver_tols, solver_args):

    options = {'ftol': 1e-16, 'maxiter': max_iters, 'disp': True, 'maxfun': 1e8}

    res = minimize(fun=obj_func, jac=obj_grad, x0=initial_value.reshape(-1),
                    args=solver_args, method='L-BFGS-B', bounds=bounds, tol=1e-10, options=options)
    solver_success = gap_func(res.x, *solver_tols, *solver_args)
    
    return SimpleNamespace(
        x=res.x,
        nit=res.nit,
        solver_success=solver_success,
        message=res.message
        )


def solve_agd(
    initial_value, 
    bounds,
    obj_func,
    obj_grad,
    gap_func,
    max_iters,
    solver_tols,
    solver_args,):
        """
        Projected gradient descent with Nesterov acceleration with forward-backward backtracking line-search on learning rate lr.
        """
        
        lr = 1e-3

        lb_list, ub_list = zip(*bounds)
        lower_bounds = create_clip_bounds(list(lb_list), -np.inf)
        upper_bounds = create_clip_bounds(list(ub_list), np.inf)

        beta, eta = 0.9, 1.1
        t = lr

        x = initial_value.copy()
        v = initial_value.copy()
        theta = 2

        for k in range(1, max_iters):

            # Check convergence
            if gap_func(x, *solver_tols, *solver_args):
                solver_success = True
                message = "Converged"
                solver_iter = k
                break

            theta = 2 / (k + 1)
            y = (1 - theta)*x + theta*v

            # Gradient Step
            grad_y = obj_grad(y, *solver_args).reshape(-1)
            v_temp = v - (t/theta) * grad_y
            v_next = np.clip(v_temp, lower_bounds, upper_bounds)

            x_next = (1 - theta)*x + theta*v_next

            # Backtracking Line Search (FISTA style)
            fx = obj_func(x_next, *solver_args)
            fy = obj_func(y, *solver_args)

            # Quadratic approximation condition
            diff = x_next - y
            quad_approx = fy + np.dot(grad_y, diff) + (np.dot(diff, diff) / (2 * t))

            if fx > quad_approx:
                # Reduce step size
                while True:
                    t *= beta
                    v_temp = v - (t/theta) * grad_y
                    v_next = np.clip(v_temp, lower_bounds, upper_bounds)
                    x_next = (1 - theta)*x + theta*v_next

                    fx = obj_func(x_next, *solver_args)
                    diff = x_next - y
                    quad_approx = fy + np.dot(grad_y, diff) + (np.dot(diff, diff) / (2 * t))
                    if fx <= quad_approx:
                        break

                v = v_next
                x = x_next
            else:
                t *= eta
                v = v_next
                x = x_next

        else:
            solver_success = False
            message = "Failure"
            solver_iter = k

        return SimpleNamespace(
            x=x,
            nit=solver_iter,
            solver_success=solver_success,
            message=message
            )