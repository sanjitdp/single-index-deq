import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from tqdm import tqdm
import os


def tanh_fn(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1 - np.tanh(z) ** 2


def sigmoid_fn(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_prime(z):
    s = sigmoid_fn(z)
    return s * (1 - s)


def relu_fn(z):
    return np.maximum(0, z)


def relu_prime(z):
    return np.where(z > 0, 1.0, np.where(z < 0, 0.0, 1.0))


def leaky_relu_fn(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)


def leaky_relu_prime(z, alpha=0.01):
    return np.where(z > 0, 1.0, np.where(z < 0, alpha, 1.0))


def gelu_fn(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))


def gelu_prime(z):
    sqrt_2_pi = np.sqrt(2 / np.pi)
    tanh_arg = sqrt_2_pi * (z + 0.044715 * z**3)
    tanh_val = np.tanh(tanh_arg)
    sech2_val = 1 - tanh_val**2

    deriv_tanh_arg = sqrt_2_pi * (1 + 3 * 0.044715 * z**2)

    return 0.5 * (1 + tanh_val) + 0.5 * z * sech2_val * deriv_tanh_arg


ACTIVATIONS = {
    "tanh": (tanh_fn, tanh_prime),
    "sigmoid": (sigmoid_fn, sigmoid_prime),
    "relu": (relu_fn, relu_prime),
    "leaky_relu": (leaky_relu_fn, leaky_relu_prime),
    "gelu": (gelu_fn, gelu_prime),
}


def get_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: {name}. Choose from {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name]


class ExplicitModel:
    def __init__(self, c_true=2.0, activation="tanh"):
        self.c_true = c_true
        self.activation = activation
        self.sigma, self.sigma_prime = get_activation(activation)
        self.theta = -0.1 + np.random.randn() * 0.01
        self.theta_history = [self.theta]

    def forward(self, x):
        return self.sigma(self.theta * x)

    def compute_gradients(self, x, y_pred, loss_grad):
        z = self.theta * x
        sigma_deriv = self.sigma_prime(z)

        df_dtheta = sigma_deriv * x

        grads = loss_grad * df_dtheta
        return grads


class ImplicitModel:
    def __init__(self, c_true=2.0, activation="tanh"):
        self.c_true = c_true
        self.activation = activation
        self.sigma, self.sigma_prime = get_activation(activation)

        self.theta = np.array([-0.1, 0.0]) + np.random.randn(2) * 0.01
        self.theta_history = [self.theta.copy()]

    def g(self, x, y):
        z = self.theta[0] * x + self.theta[1] * y
        return self.sigma(z)

    def forward(self, x):
        def h(y):
            return y - self.g(x, y)

        result = root_scalar(h, bracket=[-10, 10], method="brentq")
        return result.root

    def compute_gradients(self, x, y_star, loss_grad):
        z = self.theta[0] * x + self.theta[1] * y_star
        sigma_deriv = self.sigma_prime(z)

        dg_dy = sigma_deriv * self.theta[1]
        dg_dtheta = sigma_deriv * np.array([x, y_star])

        factor = 1.0 / (1.0 - dg_dy)
        dy_dtheta = factor * dg_dtheta

        grads = loss_grad * dy_dtheta
        return grads


def train(
    n_samples=1000,
    n_epochs=4000,
    learning_rate=0.5,
    c_true=2.0,
    model_type="implicit",
    activation="tanh",
):
    np.random.seed(42)
    x_train = np.random.randn(n_samples)

    sigma_fn, _ = get_activation(activation)
    y_train = sigma_fn(c_true * x_train)

    np.random.seed(123)
    n_test = n_samples // 5
    x_test = np.random.randn(n_test)
    y_test = sigma_fn(c_true * x_test)
    np.random.seed(42)

    if model_type == "explicit":
        model = ExplicitModel(c_true=c_true, activation=activation)
    else:
        model = ImplicitModel(c_true=c_true, activation=activation)

    train_losses = []
    test_losses = []

    pbar = tqdm(range(n_epochs), desc=f"Training ({model_type})", unit="epoch")
    for epoch in pbar:
        # Training
        total_loss = 0.0
        if isinstance(model, ExplicitModel):
            total_grads = 0.0
        else:
            total_grads = np.zeros_like(model.theta)

        for x, y_true in zip(x_train, y_train):
            y_pred = model.forward(x)

            loss = (y_pred - y_true) ** 2
            total_loss += loss

            loss_grad = 2.0 * (y_pred - y_true)
            grads = model.compute_gradients(x, y_pred, loss_grad)
            total_grads += grads

        avg_grads = total_grads / n_samples
        model.theta -= learning_rate * avg_grads
        if isinstance(model, ExplicitModel):
            model.theta_history.append(model.theta)
        else:
            model.theta_history.append(model.theta.copy())

        avg_train_loss = total_loss / n_samples
        train_losses.append(avg_train_loss)

        total_test_loss = 0.0
        for x, y_true in zip(x_test, y_test):
            y_pred = model.forward(x)
            loss = (y_pred - y_true) ** 2
            total_test_loss += loss

        avg_test_loss = total_test_loss / n_test
        test_losses.append(avg_test_loss)

        if epoch % 100 == 0:
            pbar.set_postfix(
                {
                    "train_loss": f"{avg_train_loss:.6f}",
                    "test_loss": f"{avg_test_loss:.6f}",
                }
            )

    return model, train_losses, test_losses, x_train, y_train


def plot_results(
    model, train_losses, test_losses, model_type="implicit", output_dir="."
):
    plt.style.use("math.mplstyle")

    os.makedirs(output_dir, exist_ok=True)

    plt.plot(train_losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss over epochs")
    plt.savefig(os.path.join(output_dir, f"loss_{model_type}_model.png"))
    plt.close()

    plt.figure()
    x_test = np.linspace(-2.0, 2.0, 200)
    y_true = model.sigma(model.c_true * x_test)
    label_true = f"$f(x) = \\sigma(2x)$"
    y_pred = np.array([model.forward(x) for x in x_test])

    plt.plot(x_test, y_true, "b-", label=label_true, linewidth=2)
    plt.plot(x_test, y_pred, "r--", label="Learned", linewidth=2)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title(f"Learned function")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"function_{model_type}_model.png"))
    plt.close()

    if isinstance(model, ExplicitModel):
        theta_history = np.array(model.theta_history)
        plt.figure()
        epochs = np.arange(len(theta_history))
        plt.plot(epochs, theta_history, linewidth=2)
        plt.axhline(
            y=model.c_true,
            color="r",
            linestyle="--",
            label=f"Target $\\theta = {model.c_true}$",
        )
        plt.xlabel("$t$ (Epoch)")
        plt.ylabel(r"$\theta$")
        plt.title("Parameter iterates (explicit model)")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "theta_iterates_explicit_model.png"))
        plt.close()

        plt.figure()
        plt.plot(np.abs(theta_history - model.c_true))
        plt.xlabel("Epoch")
        plt.ylabel("$|\\theta - c|$")
        plt.title(f"Distance from solution (explicit model)")
        plt.yscale("log")
        plt.savefig(
            os.path.join(output_dir, "distance_from_solution_explicit_model.png")
        )
        plt.close()
    else:
        theta_history = np.array(model.theta_history)

        plt.figure()
        iterations = np.arange(len(theta_history))
        scatter = plt.scatter(
            theta_history[:, 0],
            theta_history[:, 1],
            c=iterations,
            cmap="viridis",
            s=20,
        )
        plt.plot(
            theta_history[:, 0],
            theta_history[:, 1],
            "k-",
            alpha=0.3,
            linewidth=0.5,
        )
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$\theta_2$")
        plt.title("Gradient descent iterates")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Epoch", rotation=270, labelpad=15)
        plt.savefig(os.path.join(output_dir, "theta_iterates_implicit_model.png"))
        plt.close()

        plt.figure()
        plt.plot(theta_history[:, 0], label=r"$\theta_1$")
        plt.plot(theta_history[:, 1], label=r"$\theta_2$")
        plt.axhline(
            y=model.c_true,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"$c = {model.c_true}$",
        )
        plt.axhline(y=0, color="g", linestyle="--", alpha=0.5, label="$0$")
        plt.xlabel("Epoch")
        plt.ylabel("Parameter value")
        plt.title("Parameter components over time (implicit model)")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "theta_components_implicit_model.png"))
        plt.close()

        target = np.array([model.c_true, 0.0])
        plt.figure()
        distances = np.linalg.norm(theta_history - target, axis=1)
        plt.plot(distances)
        plt.xlabel("Epoch")
        plt.ylabel(r"\ell_2$ distance")
        plt.title(f"Distance from $(2, 0)$ over epochs")
        plt.savefig(os.path.join(output_dir, "distance_from_c0_implicit_model.png"))
        plt.close()

    plt.close("all")


def compare_models(
    n_samples=1000,
    n_epochs=4000,
    learning_rate=0.1,
    c_true=2.0,
    activation="tanh",
):
    output_dir = os.path.join("images", activation)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Activation: {activation}")
    print("=" * 60)
    print("Training Explicit Model: f_theta(x) = sigma(theta * x)")
    print("=" * 60)
    explicit_model, explicit_train_losses, explicit_test_losses, x_train, y_train = (
        train(
            n_samples=n_samples,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            c_true=c_true,
            model_type="explicit",
            activation=activation,
        )
    )
    plot_results(
        explicit_model,
        explicit_train_losses,
        explicit_test_losses,
        model_type="explicit",
        output_dir=output_dir,
    )
    print(f"\nExplicit Model - Final parameter: theta = {explicit_model.theta:.6f}")
    print(f"Target: theta = {c_true}")

    print("\n" + "=" * 60)
    print("Training Implicit Model: y = sigma(theta_1 * x + theta_2 * y)")
    print("=" * 60)
    implicit_model, implicit_train_losses, implicit_test_losses, _, _ = train(
        n_samples=n_samples,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        c_true=c_true,
        model_type="implicit",
        activation=activation,
    )
    plot_results(
        implicit_model,
        implicit_train_losses,
        implicit_test_losses,
        model_type="implicit",
        output_dir=output_dir,
    )
    print(f"\nImplicit Model - Final parameters: theta = {implicit_model.theta}")
    print(
        f"(theta_1 = {implicit_model.theta[0]:.6f}, theta_2 = {implicit_model.theta[1]:.6f})"
    )

    plt.style.use("math.mplstyle")

    plt.figure()
    plt.plot(explicit_train_losses, label="Explicit model", linewidth=2)
    plt.plot(implicit_train_losses, label="Implicit model", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train loss comparison")
    plt.legend()
    plt.yscale("log")
    plt.savefig(os.path.join(output_dir, "train_loss_comparison_both_models.png"))
    plt.close()

    plt.figure()
    plt.plot(explicit_test_losses, label="Explicit model", linewidth=2)
    plt.plot(implicit_test_losses, label="Implicit model", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.title("Test loss comparison")
    plt.legend()
    plt.yscale("log")
    plt.savefig(os.path.join(output_dir, "test_loss_comparison_both_models.png"))
    plt.close()

    plt.figure()
    x_test = np.linspace(-2.0, 2.0, 200)
    sigma_fn, _ = get_activation(activation)
    y_true = sigma_fn(c_true * x_test)
    y_explicit = np.array([explicit_model.forward(x) for x in x_test])
    y_implicit = np.array([implicit_model.forward(x) for x in x_test])

    plt.plot(x_test, y_true, "b-", label=f"$f(x) = \\sigma({c_true}x)$", linewidth=2.5)
    plt.plot(x_test, y_explicit, "r--", label="Explicit model", linewidth=2)
    plt.plot(x_test, y_implicit, "g:", label="Implicit model", linewidth=2)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Function comparison")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "function_comparison_both_models.png"))
    plt.close()

    plt.figure()
    explicit_theta_history = np.array(explicit_model.theta_history)
    implicit_theta_history = np.array(implicit_model.theta_history)

    explicit_distance = np.abs(explicit_theta_history - c_true)
    implicit_distance = np.linalg.norm(
        implicit_theta_history - np.array([c_true, 0.0]), axis=1
    )

    plt.plot(explicit_distance, label="Explicit: $|\\theta - c|$", linewidth=2)
    plt.plot(
        implicit_distance,
        label="Implicit: $\\|(\\theta_1, \\theta_2) - (c, 0)\\|_2$",
        linewidth=2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Distance from target")
    plt.title("Distance from solution comparison")
    plt.legend()
    plt.yscale("log")
    plt.savefig(os.path.join(output_dir, "distance_comparison_both_models.png"))
    plt.close()

    print("\n" + "=" * 60)
    print(f"All plots saved to: {output_dir}/")
    print("=" * 60)

    return explicit_model, implicit_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train implicit vs explicit models")
    parser.add_argument(
        "--activation",
        type=str,
        default="all",
        choices=["all"] + list(ACTIVATIONS.keys()),
        help="Activation function to use",
    )
    parser.add_argument(
        "--n_samples", type=int, default=1000, help="Number of training samples"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=400, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="Learning rate"
    )
    parser.add_argument(
        "--c_true", type=float, default=2.0, help="True parameter value"
    )

    args = parser.parse_args()

    if args.activation == "all":
        for activation in ACTIVATIONS.keys():
            print("\n" + "=" * 80)
            print(f"Running experiments with {activation} activation")
            print("=" * 80 + "\n")
            compare_models(
                n_samples=args.n_samples,
                n_epochs=args.n_epochs,
                learning_rate=args.learning_rate,
                c_true=args.c_true,
                activation=activation,
            )
    else:
        compare_models(
            n_samples=args.n_samples,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            c_true=args.c_true,
            activation=args.activation,
        )
