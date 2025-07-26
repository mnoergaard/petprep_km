import numpy as np
from scipy.integrate import cumtrapz
from scipy.stats import linregress
from abc import ABC, abstractmethod
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

class ModelPerformance:
    def __init__(self, y_true, y_pred, num_params):
        self.y_true = y_true
        self.y_pred = y_pred
        self.n = len(y_true)
        self.p = num_params

    def mse(self):
        return np.sum((self.y_true - self.y_pred) ** 2) / (self.n - self.p)

    def sigma_squared(self):
        residuals = self.y_true - self.y_pred
        return np.var(residuals, ddof=self.p)

    def log_likelihood(self):
        sigma2 = self.sigma_squared()
        return -0.5 * self.n * (np.log(2 * np.pi * sigma2) + 1)

    def aic(self):
        return -2 * self.log_likelihood() + 2 * (self.p + 1)

    def fpe(self):
        residuals = self.y_true - self.y_pred
        return np.sum(residuals ** 2) * (self.n + self.p) / (self.n - self.p)

    def coef_variation(self):
        residuals = self.y_true - self.y_pred
        mean_y_true = np.mean(self.y_true)
        return np.std(residuals, ddof=self.p) / mean_y_true if mean_y_true != 0 else np.nan

    def all_metrics(self):
        return {
            "MSE": self.mse(),
            "SigmaSqr": self.sigma_squared(),
            "LogLike": self.log_likelihood(),
            "AIC": self.aic(),
            "FPE": self.fpe(),
            "CoV": self.coef_variation(),
        }


class BaseBloodModel(ABC):
    def __init__(self, tac_times, tac_values, plasma_times, plasma_values, blood_values=None):
        self.tac_times = tac_times / 60.0  # minutes
        self.tac_values = tac_values
        self.plasma_times = plasma_times / 60.0  # minutes
        self.plasma_values = plasma_values
        self.blood_values = blood_values if blood_values is not None else plasma_values

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def visualize_fit(self, output_path, region_name):
        pass


class MA1Model(BaseBloodModel):
    parameters = ["VT", "intercept", "coef_X2", "MSE", "SigmaSqr", "LogLike", "AIC", "FPE"]

    def __init__(self, tac_times, tac_values, plasma_times, plasma_values, t_star):
        super().__init__(tac_times, tac_values, plasma_times, plasma_values)
        self.t_star = t_star

    def fit(self):
        # Convert time to minutes
        tac_minutes = self.tac_times
        plasma_interp = np.interp(tac_minutes, self.plasma_times, self.plasma_values)

        auc_input = cumtrapz(plasma_interp, tac_minutes, initial=0)
        auc_pet = cumtrapz(self.tac_values, tac_minutes, initial=0)

        mask = tac_minutes >= self.t_star

        X = np.column_stack((auc_input[mask], auc_pet[mask]))
        Y = self.tac_values[mask]

        model = sm.OLS(Y, X)
        results = model.fit()

        b1, b2 = results.params
        y_pred = results.fittedvalues
        residuals = Y - y_pred
        n = len(Y)
        p = 2

        mean_y = np.mean(Y)
        cov = np.std(residuals, ddof=p) / mean_y if mean_y != 0 else np.nan
        mse = np.sum(residuals ** 2) / (n - p)
        sigma_squared = np.var(residuals, ddof=p)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma_squared) - 0.5 * np.sum(residuals ** 2) / sigma_squared
        aic = -2 * log_likelihood + 2 * (p + 1)
        fpe = np.sum(residuals ** 2) * (n + p) / (n - p)

        VT = -b1 / b2 if b2 != 0 else np.nan
        intercept = 1.0 / b2 if b2 != 0 else np.nan

        self.fit_result = {
            "VT": VT,
            "intercept": intercept,
            "coef_X2": b1,
            "MSE": mse,
            "SigmaSqr": sigma_squared,
            "LogLike": log_likelihood,
            "AIC": aic,
            "FPE": fpe,
            "CoV": cov
        }
        return self.fit_result

    def visualize_fit(self, output_path, region_name):
        # Convert time to minutes
        tac_minutes = self.tac_times
        plasma_interp = np.interp(tac_minutes, self.plasma_times, self.plasma_values)

        auc_input = cumtrapz(plasma_interp, tac_minutes, initial=0)
        auc_pet = cumtrapz(self.tac_values, tac_minutes, initial=0)

        mask = tac_minutes >= self.t_star

        b1 = self.fit_result["coef_X2"]
        b2 = -b1 / self.fit_result["VT"] if self.fit_result["VT"] != 0 else np.nan
        fit_line = b1 * auc_input + b2 * auc_pet

        plt.figure(figsize=(8, 4))
        plt.plot(tac_minutes, self.tac_values, 'ko', label='TAC')
        plt.plot(tac_minutes[mask], self.tac_values[mask], 'ro', label='Fitting Points')
        plt.plot(tac_minutes, fit_line, 'r--', label='MA1 Fit')
        plt.title(region_name)
        plt.xlabel("Time (min)")
        plt.ylabel("Radioactivity Concentration")
        plt.legend()
        plt.annotate(
            f"VT = {self.fit_result['VT']:.2f}\nIntercept = {self.fit_result['intercept']:.2f}\nCoV = {self.fit_result['CoV']:.4f}",
            xy=(0.6, 0.1), xycoords='axes fraction')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

class LoganModel(BaseBloodModel):
    parameters = ["VT", "Kappa2", "VT_var", "intercept", "R_squared", "MSE", "SigmaSqr", "LogLike", "AIC", "FPE", "CoV"]

    def __init__(self, tac_times, tac_values, plasma_times, plasma_values, 
                 blood_values=None, t_star=None):
        super().__init__(tac_times, tac_values, plasma_times, plasma_values, blood_values)
        self.t_star = t_star

    def fit(self):
        # Convert time from seconds to minutes for modeling consistency
        tac_minutes = self.tac_times
        plasma_interp = np.interp(tac_minutes, self.plasma_times, self.plasma_values)

        integral_tac = cumtrapz(self.tac_values, tac_minutes, initial=0)
        integral_plasma = cumtrapz(plasma_interp, tac_minutes, initial=0)

        mask = tac_minutes >= self.t_star
        x = integral_plasma[mask] / self.tac_values[mask]
        y = integral_tac[mask] / self.tac_values[mask]

        # Linear regression using statsmodels to get standard errors
        X_design = sm.add_constant(x)
        glm_model = sm.WLS(y, X_design)
        glm_results = glm_model.fit()

        intercept, VT = glm_results.params
        cov_matrix = glm_results.cov_params()
        VT_var = cov_matrix[1, 1]

        y_pred = glm_results.fittedvalues
        residuals = y - y_pred
        n = len(y)
        p = 2  # parameters: intercept and VT

        mean_y = np.mean(y)
        cov = np.std(residuals, ddof=p) / mean_y if mean_y != 0 else np.nan
        mse = np.sum(residuals ** 2) / (n - p)
        sigma_squared = np.var(residuals, ddof=p)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma_squared) - 0.5 * np.sum(residuals ** 2) / sigma_squared
        aic = -2 * log_likelihood + 2 * (p + 1)  # 2 parameters + noise variance
        fpe = np.sum(residuals ** 2) * (n + p) / (n - p)
        r_squared = glm_results.rsquared

        Kappa2 = -1 / intercept if intercept != 0 else np.nan

        self.fit_result = {
            "VT": VT,
            "Kappa2": Kappa2,
            "VT_var": VT_var,
            "intercept": intercept,
            "R_squared": r_squared,
            "MSE": mse,
            "SigmaSqr": sigma_squared,
            "LogLike": log_likelihood,
            "AIC": aic,
            "FPE": fpe,
            "CoV": cov
        }
        return self.fit_result

    def visualize_fit(self, output_path, region_name):
        # Convert time to minutes
        tac_minutes = self.tac_times
        plasma_interp = np.interp(tac_minutes, self.plasma_times, self.plasma_values)

        integral_tac = cumtrapz(self.tac_values, tac_minutes, initial=0)
        integral_plasma = cumtrapz(plasma_interp, tac_minutes, initial=0)

        mask = tac_minutes >= self.t_star
        x = integral_plasma[mask] / self.tac_values[mask]
        y = integral_tac[mask] / self.tac_values[mask]

        X_design = sm.add_constant(x)
        y_pred = X_design @ np.array([self.fit_result["intercept"], self.fit_result["VT"]])

        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, 'ko', label='Data')
        plt.plot(x, y_pred, 'r--', label='Logan Fit')
        plt.xlabel('∫Cp(t)/Ct(t) [min]')
        plt.ylabel('∫Ct(t)/Ct(t) [min]')
        plt.title(f'Logan Plot - {region_name}')
        plt.annotate(
            f"VT = {self.fit_result['VT']:.2f}\nKappa2 = {self.fit_result['Kappa2']:.4f}\nt_star = {self.t_star:.2f} min\nCoV = {self.fit_result['CoV']:.4f}",
            xy=(0.65, 0.1), xycoords='axes fraction'
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

class TwoTCMModel(BaseBloodModel):
    parameters = ["K1", "k2", "k3", "k4", "vB", "VT", "CoV"]

    def __init__(self, tac_times, tac_values, plasma_times, plasma_values, blood_values=None,
                 bounds_lower=None, bounds_upper=None, vB_fixed=None, inpshift=0.0, fit_end_time=None,
                 n_iterations=50):

        super().__init__(tac_times, tac_values, plasma_times, plasma_values, blood_values)
        self.vB_fixed = vB_fixed
        self.inpshift = inpshift
        self.fit_end_time = fit_end_time or self.tac_times[-1]
        self.n_iterations = n_iterations

        if self.vB_fixed is not None:
            self.bounds_lower = bounds_lower or [0.0001, 0.0001, 0.0001, 0.0001]
            self.bounds_upper = bounds_upper or [1.0, 0.5, 0.5, 0.5]
        else:
            self.bounds_lower = bounds_lower or [0.001, 0.001, 0.001, 0.001, 0.01]
            self.bounds_upper = bounds_upper or [1.0, 0.5, 0.5, 0.5, 0.1]

    def fit(self):
        mask = self.tac_times <= self.fit_end_time
        t_pet = self.tac_times[mask]
        tac_pet = self.tac_values[mask]

        plasma_shifted_times = self.plasma_times + self.inpshift
        Cp = interp1d(plasma_shifted_times, self.plasma_values, bounds_error=False, fill_value="extrapolate")(t_pet)
        Cb = interp1d(plasma_shifted_times, self.blood_values, bounds_error=False, fill_value="extrapolate")(t_pet)

        best_fit = None
        min_cost = np.inf

        for _ in range(self.n_iterations):
            x0 = np.random.uniform(self.bounds_lower, self.bounds_upper)
            res = least_squares(self._residuals, x0, bounds=(self.bounds_lower, self.bounds_upper),
                                args=(t_pet, Cp, Cb, tac_pet))
            if res.cost < min_cost:
                min_cost = res.cost
                best_fit = res.x

        if self.vB_fixed is not None:
            K1, k2, k3, k4 = best_fit
            vB = self.vB_fixed
        else:
            K1, k2, k3, k4, vB = best_fit

        VT = (K1 / k2) * (1 + k3 / k4)

        residuals = self._residuals(best_fit, t_pet, Cp, Cb, tac_pet)
        mean_tac_pet = np.mean(tac_pet)
        cov = np.std(residuals, ddof=len(best_fit)) / mean_tac_pet if mean_tac_pet != 0 else np.nan

        self.fit_result = {
            "K1": K1, "k2": k2, "k3": k3, "k4": k4,
            "vB": vB, "VT": VT,
            "CoV": cov
        }

        return self.fit_result

    def _residuals(self, params, t, Cp, Cb, tac_pet):
        if self.vB_fixed is not None:
            K1, k2, k3, k4 = params
            vB = self.vB_fixed
        else:
            K1, k2, k3, k4, vB = params

        Ct_model = self._simulate_2tcm(t, Cp, Cb, K1, k2, k3, k4, vB)
        return Ct_model - tac_pet

    def _simulate_2tcm(self, t, Cp, Cb, K1, k2, k3, k4, vB):
        dt = np.diff(t, prepend=0)
        C1, C2, Ct = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)

        for i in range(1, len(t)):
            dC1 = dt[i] * (K1 * Cp[i] - (k2 + k3) * C1[i-1] + k4 * C2[i-1])
            dC2 = dt[i] * (k3 * C1[i-1] - k4 * C2[i-1])
            C1[i] = C1[i-1] + dC1
            C2[i] = C2[i-1] + dC2
            Ct[i] = (1 - vB) * (C1[i] + C2[i]) + vB * Cb[i]

        return Ct

    def visualize_fit(self, output_path, region_name):
        mask = self.tac_times <= self.fit_end_time
        t_pet = self.tac_times[mask]

        plasma_shifted_times = self.plasma_times + self.inpshift
        Cp = interp1d(plasma_shifted_times, self.plasma_values, bounds_error=False, fill_value="extrapolate")(t_pet)
        Cb = interp1d(plasma_shifted_times, self.blood_values, bounds_error=False, fill_value="extrapolate")(t_pet)

        fit_curve = self._simulate_2tcm(
            t_pet, Cp, Cb,
            K1=self.fit_result["K1"],
            k2=self.fit_result["k2"],
            k3=self.fit_result["k3"],
            k4=self.fit_result["k4"],
            vB=self.fit_result["vB"]
        )

        plt.figure(figsize=(8, 4))
        plt.plot(t_pet, self.tac_values[mask], 'ko', label='Measured TAC')
        plt.plot(t_pet, fit_curve, 'r-', label='2TCM Fit')
        plt.title(region_name)
        plt.xlabel("Time (min)")
        plt.ylabel("Radioactivity Concentration")
        plt.annotate(
            f"VT = {self.fit_result['VT']:.2f}\nCoV = {self.fit_result['CoV']:.4f}\nvB = {self.fit_result['vB']:.4f}",
            xy=(0.65, 0.5), xycoords='axes fraction'
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

class OneTCMModel(BaseBloodModel):
    parameters = ["K1", "k2", "vB", "VT", "CoV"]

    def __init__(self, tac_times, tac_values, plasma_times, plasma_values, blood_values=None,
                 bounds_lower=None, bounds_upper=None, vB_fixed=None, fit_end_time=None,
                 n_iterations=50):
        super().__init__(tac_times, tac_values, plasma_times, plasma_values, blood_values)
        self.bounds_lower = bounds_lower or [0.0001, 0.0001, 0.01]
        self.bounds_upper = bounds_upper or [1.0, 0.5, 0.1]
        self.vB_fixed = vB_fixed
        self.fit_end_time = fit_end_time or self.tac_times[-1]
        self.n_iterations = n_iterations

    def fit(self):
        mask = self.tac_times <= self.fit_end_time
        tac_minutes = self.tac_times[mask]
        tac_values = self.tac_values[mask]

        plasma_minutes = self.plasma_times
        ca = interp1d(plasma_minutes, self.plasma_values, fill_value="extrapolate")
        cb = interp1d(plasma_minutes, self.blood_values, fill_value="extrapolate")

        def residuals(params):
            if self.vB_fixed is not None:
                K1, k2 = params
                vB = self.vB_fixed
            else:
                K1, k2, vB = params
            Ct_pred = self._simulate_1tcm(tac_minutes, ca, cb, K1, k2, vB)
            return Ct_pred - tac_values

        best_fit = None
        min_cost = np.inf

        if self.vB_fixed is not None:
            bounds_lower = self.bounds_lower[:2]
            bounds_upper = self.bounds_upper[:2]
        else:
            bounds_lower = self.bounds_lower
            bounds_upper = self.bounds_upper

        for _ in range(self.n_iterations):
            x0 = np.random.uniform(bounds_lower, bounds_upper)
            res = least_squares(residuals, x0, bounds=(bounds_lower, bounds_upper))
            if res.cost < min_cost:
                min_cost = res.cost
                best_fit = res.x

        if self.vB_fixed is not None:
            K1, k2 = best_fit
            vB = self.vB_fixed
        else:
            K1, k2, vB = best_fit

        VT = K1 / k2 if k2 != 0 else np.nan

        residual_values = residuals(best_fit)
        mean_tac_pet = np.mean(tac_values)
        cov = np.std(residual_values, ddof=len(best_fit)) / mean_tac_pet if mean_tac_pet != 0 else np.nan

        self.fit_result = {"K1": K1, "k2": k2, "vB": vB, "VT": VT, "CoV": cov}
        return self.fit_result

    def visualize_fit(self, output_path, region_name):
        mask = self.tac_times <= self.fit_end_time
        tac_minutes = self.tac_times[mask]

        plasma_minutes = self.plasma_times
        ca = interp1d(plasma_minutes, self.plasma_values, fill_value="extrapolate")
        cb = interp1d(plasma_minutes, self.blood_values, fill_value="extrapolate")

        fit_curve = self._simulate_1tcm(tac_minutes, ca, cb, K1=self.fit_result["K1"], k2=self.fit_result["k2"], vB=self.fit_result["vB"])

        plt.figure(figsize=(8, 4))
        plt.plot(tac_minutes, self.tac_values[mask], 'ko', label='Measured TAC')
        plt.plot(tac_minutes, fit_curve, 'r--', label='1TCM Fit')
        plt.title(region_name)
        plt.xlabel("Time (min)")
        plt.ylabel("Radioactivity Concentration")
        plt.annotate(
            f"VT = {self.fit_result['VT']:.2f}\nCoV = {self.fit_result['CoV']:.4f}\nvB = {self.fit_result['vB']:.4f}",
            xy=(0.65, 0.5), xycoords='axes fraction'
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _simulate_1tcm(self, t, ca_func, cb_func, K1, k2, vB):
        dt = np.diff(t, prepend=0)
        Ct = np.zeros_like(t)
        C1 = 0

        for i in range(1, len(t)):
            dC1 = dt[i] * (K1 * ca_func(t[i]) - k2 * C1)
            C1 += dC1
            Ct[i] = (1 - vB) * C1 + vB * cb_func(t[i])

        return Ct