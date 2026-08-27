import numpy as np

def f_pred(X_train, X_test, kernel, Ahat, chol_gram, chol_gram_inference, use_low_rank, method_low_rank):
    """
    Prediction of a kSoS function on test samples X_test, with computation of empirical feature map phi_test (Equation (4)) 
    and acceleration of kSoS formula (Equation (8)). The predictions f_test are clipped due to possible numerical errors.
    """
    k_train_test = kernel(X_train, X_test)
    match use_low_rank:
        case False:
            phi_test = np.linalg.solve(chol_gram.T, k_train_test)
        case True:
            match method_low_rank:
                case "svd" | "rsvd":
                    phi_test = chol_gram_inference @ k_train_test
                case "nystrom":
                    phi_test = np.linalg.solve(chol_gram.T, k_train_test)
    f_test = np.einsum('ji,jk,ki->i', phi_test, Ahat, phi_test, optimize=True)
    return np.maximum(f_test, 1e-12)

