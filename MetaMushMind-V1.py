import numpy as np
import pandas as pd
import random
import logging
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Any
from scipy.stats import randint, uniform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedMycoOptimization:
    def __init__(self, n_agents: int, decay: float = 0.1, alpha: float = 1, beta: float = 2,
                 adaptation_rate: float = 0.05, exploration_rate: float = 0.1):
        self.n_agents = n_agents
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.adaptation_rate = adaptation_rate
        self.exploration_rate = exploration_rate
        self.best_params = None
        self.best_score = -np.inf
        self.pheromones = {}
        self.learning_rate = 0.01
        self.history = []

    def initialize_pheromones(self, param_space: Dict[str, List[Any]]) -> None:
        self.pheromones = {param: np.ones(len(values)) for param, values in param_space.items()}

    def choose_params(self, param_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        return {
            param: (random.choice(values) if np.random.rand() < self.exploration_rate
                    else random.choices(values, weights=self.pheromones[param]**self.alpha / np.sum(self.pheromones[param]**self.alpha), k=1)[0])
            for param, values in param_space.items()
        }

    def update_pheromones(self, param_space: Dict[str, List[Any]], chosen_params: Dict[str, Any], score: float) -> None:
        for param, values in param_space.items():
            index = values.index(chosen_params[param])
            self.pheromones[param][index] += score * self.beta
            self.pheromones[param] = self.pheromones[param] * (1 - self.decay) + self.decay

    def adjust_hyperparameters(self, score: float) -> None:
        if score > self.best_score:
            self.learning_rate *= (1 + self.adaptation_rate)
            self.exploration_rate *= (1 - self.adaptation_rate)
        else:
            self.learning_rate *= (1 - self.adaptation_rate)
            self.exploration_rate *= (1 + self.adaptation_rate)
        
        self.learning_rate = np.clip(self.learning_rate, 0.0001, 0.1)
        self.exploration_rate = np.clip(self.exploration_rate, 0.01, 0.5)

    def evaluate_model(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        model = MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'],
                              activation=params['activation'],
                              solver=params['solver'],
                              learning_rate_init=self.learning_rate,
                              max_iter=1000,
                              early_stopping=True,
                              n_iter_no_change=20,
                              validation_fraction=0.1)
        
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        accuracy = np.mean(scores)
        
        model.fit(X, y)
        y_pred = model.predict(X)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted', zero_division=1)
        
        return accuracy, {'precision': precision, 'recall': recall, 'f1': f1}

    def optimize(self, param_space: Dict[str, List[Any]], X: np.ndarray, y: np.ndarray, iterations: int = 100) -> Tuple[Dict[str, Any], float]:
        self.initialize_pheromones(param_space)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for iteration in range(iterations):
            logging.info(f"Iteration {iteration + 1}/{iterations}")
            
            results = Parallel(n_jobs=-1)(
                delayed(self.agent_optimization)(param_space, X_scaled, y)
                for _ in range(self.n_agents)
            )

            for chosen_params, score, metrics in results:
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = chosen_params
                    logging.info(f"New best score: {score:.4f}, Params: {chosen_params}")
                
                self.update_pheromones(param_space, chosen_params, score)
                self.adjust_hyperparameters(score)
                
                self.history.append({
                    'iteration': iteration,
                    'params': chosen_params,
                    'score': score,
                    'metrics': metrics
                })

            if self.check_convergence():
                logging.info(f"Converged after {iteration + 1} iterations")
                break

        return self.best_params, self.best_score

    def agent_optimization(self, param_space: Dict[str, List[Any]], X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
        chosen_params = self.choose_params(param_space)
        score, metrics = self.evaluate_model(X, y, chosen_params)
        return chosen_params, score, metrics

    def check_convergence(self, window: int = 10, threshold: float = 0.0005) -> bool:
        if len(self.history) < window:
            return False
        recent_scores = [entry['score'] for entry in self.history[-window:]]
        return np.std(recent_scores) < threshold

    def get_optimization_summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.history)
        df['learning_rate'] = [self.learning_rate] * len(df)
        df['exploration_rate'] = [self.exploration_rate] * len(df)
        return df

class MetaMushMind:
    def __init__(self, base_param_space: Dict[str, List[Any]], iterations: int = 10):
        self.base_param_space = base_param_space
        self.iterations = iterations
        self.best_mushmind = None
        self.best_score = -np.inf
        self.history = []

    def optimize(self, X: np.ndarray, y: np.ndarray):
        mushmind_param_space = {
            'n_agents': randint(5, 30),
            'decay': uniform(0.05, 0.25),
            'alpha': uniform(0.5, 2.0),
            'beta': uniform(1.0, 4.0),
            'adaptation_rate': uniform(0.01, 0.2),
            'exploration_rate': uniform(0.05, 0.3)
        }

        for i in range(self.iterations):
            logging.info(f"MetaMushMind Iteration {i + 1}/{self.iterations}")
            mushmind_params = {k: v.rvs() for k, v in mushmind_param_space.items()}
            
            optimizer = AdvancedMycoOptimization(**mushmind_params)
            best_params, best_score = optimizer.optimize(self.base_param_space, X, y, iterations=50)
            
            self.history.append({
                'iteration': i,
                'mushmind_params': mushmind_params,
                'best_params': best_params,
                'best_score': best_score
            })
            
            if best_score > self.best_score:
                self.best_score = best_score
                self.best_mushmind = optimizer
                logging.info(f"New best MetaMushMind score: {best_score:.4f}")
                logging.info(f"Best MushMind params: {mushmind_params}")
                logging.info(f"Best model params: {best_params}")

        return self.best_mushmind.best_params, self.best_score

    def get_optimization_summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load and prepare data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # Define parameter space
    param_space = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (50, 100, 50), (100, 100, 50)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam']
    }

    # Create and run MetaMushMind
    meta_mushmind = MetaMushMind(param_space, iterations=100)
    best_params, best_score = meta_mushmind.optimize(X_train, y_train)

    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")

    # Evaluate on test set
    final_model = MLPClassifier(**best_params, max_iter=1000)
    final_model.fit(X_train, y_train)
    test_score = final_model.score(X_test, y_test)
    print(f"Test accuracy: {test_score:.4f}")