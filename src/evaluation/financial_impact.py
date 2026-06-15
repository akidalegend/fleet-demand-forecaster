class FinancialImpactEvaluator:
    def __init__(self, cost_per_idle_hour=15.00, opportunity_cost_lost_ride=22.50):
        self.cost_idle = cost_per_idle_hour
        self.cost_lost = opportunity_cost_lost_ride

    def calculate_deadweight_loss(self, y_true, y_pred):
        errors = y_pred - y_true
        # Over-forecasting: Drivers sit idle
        over_forecast_cost = np.sum(np.where(errors > 0, errors * (15/60) * self.cost_idle, 0))
        # Under-forecasting: Lost market demand
        under_forecast_cost = np.sum(np.where(errors < 0, np.abs(errors) * self.cost_lost, 0))
        
        return over_forecast_cost + under_forecast_cost