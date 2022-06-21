from genetic_selection import GeneticSelectionCV


class Selector:
    def __init__(self, estimator=None, n_jobs: int = 2):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.selector_model = None
        self.select_estimator()

    def select_estimator(self):

        self.selector_model = GeneticSelectionCV(
            self.estimator, cv=5, verbose=0,
            scoring="f1_weighted", max_features=50,
            n_population=50, crossover_proba=0.5,
            mutation_proba=0.2, n_generations=50,
            crossover_independent_proba=0.5,
            mutation_independent_proba=0.04,
            tournament_size=3, n_gen_no_change=10,
            caching=True, n_jobs=self.n_jobs)
