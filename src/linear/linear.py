import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from tqdm import tqdm


class ImplicitModel:
    def __init__(self, intercept=True):
        self.intercept = intercept
        if intercept:
            self.theta = np.random.randn(3) * 0.1
        else:
            self.theta = np.random.randn(2) * 0.1 + np.array([0, 1])
        self.theta_history = [self.theta.copy()]

    def g(self, x, y):
        if self.intercept:
            return self.theta[0] + self.theta[1] * x + self.theta[2] * y
        else:
            return self.theta[0] * x + self.theta[1] * y

    def forward(self, x):
        def h(y):
            return y - self.g(x, y)

        result = root_scalar(h, bracket=[-100, 100], method="brentq")
        return result.root

    def compute_gradients(self, x, y_star, loss_grad):
        if self.intercept:
            dg_dy = self.theta[2]
            dg_dtheta = np.array([1.0, x, y_star])
        else:
            dg_dy = self.theta[1]
            dg_dtheta = np.array([x, y_star])

        factor = 1.0 / (1.0 - dg_dy)
        dy_dtheta = factor * dg_dtheta

        grads = loss_grad * dy_dtheta
        return grads


def train(n_samples=1000, n_epochs=400, learning_rate=0.01, intercept=True):
    np.random.seed(42)
    x_train = np.random.uniform(0.0, 1.0, n_samples)

    c_true = 2.0
    d_true = 1.0 if intercept else 0.0
    y_train = c_true * x_train + d_true

    model = ImplicitModel(intercept=intercept)
    losses = []

    pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in pbar:
        total_loss = 0.0
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
        model.theta_history.append(model.theta.copy())

        avg_loss = total_loss / n_samples
        losses.append(avg_loss)

        if epoch % 100 == 0:
            pbar.set_postfix({"loss": f"{avg_loss:.6f}"})

    return model, losses, x_train, y_train


def plot_results(model, losses):
    plt.style.use("math.mplstyle")

    plt.plot(losses)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")

    plt.savefig("loss.png")
    plt.close()

    plt.figure()

    x_test = np.linspace(0.0, 1.0, 200)
    if model.intercept:
        y_true = 2.0 * x_test + 1.0
        label_true = "$f(x) = 2x + 1$"
    else:
        y_true = 2.0 * x_test
        label_true = "$f(x) = 2x$"
    y_pred = np.array([model.forward(x) for x in x_test])

    plt.plot(x_test, y_true, "b-", label=label_true, linewidth=2)
    plt.plot(x_test, y_pred, "r--", label="Learned", linewidth=2)

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Learned function")
    plt.legend()

    plt.savefig("learned.png")
    plt.close()

    theta_history = np.array(model.theta_history)

    if model.intercept:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        iterations = np.arange(len(theta_history))
        scatter = ax.scatter(
            theta_history[:, 0],
            theta_history[:, 1],
            theta_history[:, 2],
            c=iterations,
            cmap="viridis",
            s=10,
        )
        ax.plot(
            theta_history[:, 0],
            theta_history[:, 1],
            theta_history[:, 2],
            "k-",
            alpha=0.3,
            linewidth=0.5,
        )
        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$\theta_1$")
        ax.set_zlabel(r"$\theta_2$")
        ax.set_title("Gradient descent iterates")
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label("Epoch", rotation=270, labelpad=15)
        plt.savefig("iterates.png")
        plt.close()

        plt.figure()
        distances = np.linalg.norm(theta_history - np.array([0, 0, 1]), axis=1)
        plt.plot(distances)
        plt.xlabel("Epoch")
        plt.ylabel("$\\ell_2$ distance")
        plt.title("Distance from trivial solution over epochs")
        plt.savefig("distance_trivial.png")
    else:
        plt.figure()

        xi = 2.0
        alpha_vals = np.linspace(-2, 3, 100)
        line_x = alpha_vals * xi
        line_y = 1 - alpha_vals
        plt.plot(
            line_x,
            line_y,
            "g--",
            linewidth=1,
            label=r"$(\alpha \xi, 1-\alpha)$",
        )

        plt.plot(0, 1, "r*", markersize=8, label=r"Trivial solution $\theta = (0, 1)$")

        x_min = min(theta_history[:, 0].min(), 0) - 0.05
        x_max = max(theta_history[:, 0].max(), 0) + 0.05
        y_min = min(theta_history[:, 1].min(), 1) - 0.05
        y_max = max(theta_history[:, 1].max(), 1) + 0.05
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

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
        plt.legend()
        cbar = plt.colorbar(scatter)
        cbar.set_label("Epoch", rotation=270, labelpad=15)
        plt.savefig("iterates.png")
        plt.close()

        plt.figure()
        plt.plot(np.linalg.norm(theta_history - np.array([0, 1]), axis=1))
        plt.xlabel("Epoch")
        plt.ylabel("$\\ell_2$ distance")
        plt.title("Distance from $(0, 1)$ over epochs")
        plt.savefig("distance_trivial.png")

    plt.close()


if __name__ == "__main__":
    model, losses, x_train, y_train = train(intercept=False, n_epochs=200)
    plot_results(model, losses)

    print(f"\nFinal parameters: {model.theta}")
