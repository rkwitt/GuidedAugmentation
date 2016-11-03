class meta:
    def __init__( self ):
        self.object_name = None
        self.covariate_lo = None
        self.covariate_hi = None
        self.covariate_targets = []

        self.feature_regression_cv_files = []
        self.feature_regression_pretrained_model = None
        self.feature_regression_train_file = None
        self.feature_regression_trained_regressors = []
