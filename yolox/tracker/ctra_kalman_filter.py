import numpy as np

# # [x, y, v, θ, dv, ω, a, h, da, dh]
# # [x, y, a, h] - Use the position noise
# # [θ, v, dv, da, dh, ω] - Use the velocity noise
# # [x, y, a, h, θ, v, dv, da, dh, ω] 
# class OldCTRAKalmanFilter:
#     """
#     CTRA (Constant Turn Rate and Acceleration) Kalman Filter for tracking objects in 2D space.
#     Tracks a 6D state: [x, y, a, h, θ, v, dv, da, dh, ω], where:
#       - x, y : Position in 2D space (0, 1)
#       - a, h : Aspect ratio and height (2, 3)
#       - θ    : Heading angle (orientation in radians) (4)
#       - v    : Velocity (5)
#       - dv   : Acceleration (rate of change of velocity) (6)
#       - da   : Rate of change of aspect ratio (7)
#       - dh   : Rate of change of height (8)
#       - ω    : Turn rate (rate of change of heading angle) (9)
#     """

#     def __init__(self):
#         dt = 1.0  # Time step (in seconds)

#         # Motion matrix (F): Models how the state evolves over time
#         # x_new = x + v * dt, y_new = y + v * dt, etc.
#         #self._motion_mat = np.eye(6)  # Initialize as an identity matrix
#         #self._motion_mat[0, 2] = dt  # x depends on velocity (v)
#         #self._motion_mat[1, 2] = dt  # y depends on velocity (v)
#         #self._motion_mat[2, 4] = dt  # v depends on acceleration (a)
#         #self._motion_mat[3, 5] = dt  # θ depends on turn rate (ω)

#         # Update matrix (H): Maps the full state to the observed state
#         # For example, maps [x, y, v, θ, dv, ω, a, h, da, dh] to observed [x, y, a, h]
#         self._update_mat = np.zeros((4, 10))
#         self._update_mat[0, 0] = 1  # x
#         self._update_mat[1, 1] = 1  # y
#         self._update_mat[2, 2] = 1  # a
#         self._update_mat[3, 3] = 1  # h

#         # Process noise parameters
#         # These represent the expected uncertainty in the object's motion
#         self._std_weight_position = 1. / 20  # Position noise scaling factor
#         self._std_weight_velocity = 1. / 160  # Velocity noise scaling factor

#         # Measurement noise: Uncertainty in the measurements (e.g., x, y from sensors)
#         self._measurement_noise = 0.1  # Standard deviation of measurement noise



#     def initiate(self, measurement):
#         mean = np.zeros(6)  # [x, y, v, θ, a, ω]
#         mean[:2] = measurement[:2]  # x, y
#         mean[3] = measurement[2]  # θ
#         covariance = np.diag([2 * self._std_weight_position] * 2 +
#                              [1.0, 0.5, 0.1, 0.1]) ** 2
#         return mean, covariance

#     def predict(self, mean, covariance, dt=1.0):
#         mean[0] += mean[2] * dt * np.cos(mean[3])  # x
#         mean[1] += mean[2] * dt * np.sin(mean[3])  # y
#         mean[2] += mean[4] * dt  # velocity
#         mean[3] += mean[5] * dt  # heading angle
#         motion_cov = np.diag([self._std_weight_position] * 6)
#         covariance += motion_cov
#         return mean, covariance

#     def predict_state(self, state, dt=1.0):
#         print("Debug Info:")
#         predicted_state = state.copy()
#         predicted_state[0] += predicted_state[2] * dt * np.cos(predicted_state[3])
#         predicted_state[1] += predicted_state[2] * dt * np.sin(predicted_state[3])
#         predicted_state[2] += predicted_state[4] * dt
#         predicted_state[3] += predicted_state[5] * dt
#         return predicted_state

#     def update(self, mean, covariance, measurement):
#         H = np.array([[1, 0, 0, 0, 0, 0],
#                       [0, 1, 0, 0, 0, 0]])
#         print(H.shape)
#         #R = np.eye(2) * self._measurement_noise
#         #print(R.shape)

#         print("Debug Info:")
#         print("Mean shape:", mean.shape)
#         print("Covariance shape:", covariance.shape)
#         print("Measurement shape:", measurement.shape)

#         # Kalman gain
#         S = H @ covariance @ H.T + R
#         K = covariance @ H.T @ np.linalg.inv(S)

#         # Measurement residual
#         residual = measurement - H @ mean
#         updated_mean = mean + K @ residual
#         updated_covariance = (np.eye(mean.shape[0]) - K @ H) @ covariance

#         return updated_mean, updated_covariance

#     def multi_predict(self, means, covariances, dt=1.0):
#         motion_mats = np.tile(self._motion_mat, (len(means), 1, 1))
#         motion_mats[:, 0, 2] = dt * np.cos(means[:, 3])
#         motion_mats[:, 1, 2] = dt * np.sin(means[:, 3])
#         motion_mats[:, 2, 4] = dt
#         motion_mats[:, 3, 5] = dt

#         means = np.einsum('ijk,ik->ij', motion_mats, means)
#         covariances = np.einsum('ijk,ikl->ijl', motion_mats, covariances)
#         covariances = np.einsum('ijl,ikl->ijk', covariances, motion_mats)
#         return means, covariances


# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}




class CTRAKalmanFilter(object):
    """
    CTRA (Constant Turn Rate and Acceleration) Kalman Filter for tracking objects in 2D space.
    Tracks a 10D state: [x, y, a, h, θ, v, dv, da, dh, ω], where:
      - x, y : Position in 2D space (0, 1)
      - a, h : Aspect ratio and height (2, 3)
      - θ    : Heading angle (orientation in radians) (4)
      - v    : Velocity (5)
      - dv   : Acceleration (rate of change of velocity) (6)
      - da   : Rate of change of aspect ratio (7)
      - dh   : Rate of change of height (8)
      - ω    : Turn rate (rate of change of heading angle) (9)
    """

    def __init__(self):
        self.ndim, self.ntotal, self.dt = 4, 10, 1.

        self._update_mat = np.eye(self.ndim, self.ntotal)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (10 dimensional) and covariance matrix (10x10
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros(self.ntotal - self.ndim)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 10 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 10x10 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        eps = 1e-20
        [x, y, a, h, θ, v, dv, da, dh, ω] = mean
        ω = ω
        mean = np.zeros(self.ntotal)
        v_w_ratio = v / (ω + eps)
        #x
        mean[0] = x + (v_w_ratio * (np.sin(θ + (ω * self.dt)) - np.sin(θ)))
        #y
        mean[1] = y + (v_w_ratio * (-np.cos(θ + (ω * self.dt)) + np.cos(θ)))
        #a
        mean[2] = a + (da * self.dt)
        #h
        mean[3] = h + (dh * self.dt)
        #θ
        mean[4] = θ + (ω * self.dt)
        #v
        mean[5] = v + (dv * self.dt)
        #dv
        mean[6] = dv
        #da
        mean[7] = da
        #dh
        mean[8] = dh
        #ω
        mean[9] = ω

        # Generate the used jacobian matrix
        # We start with an identity matrix which covers:
        #mean[0] = x
        #mean[1] = y
        #mean[2] = a
        #mean[3] = h
        #mean[4] = θ
        #mean[5] = v
        #mean[6] = dv
        #mean[7] = da
        #mean[8] = dh
        #mean[9] = ω
        #The following terms still need to be filled out to complete the jacobian:
        #mean[0] = (v_w_ratio * (np.sin(θ + (ω * self.dt)) - np.sin(θ)))
        #mean[1] = (v_w_ratio * (-np.cos(θ + (ω * self.dt)) + np.cos(θ)))
        #mean[2] = (da * self.dt)
        #mean[3] = (dh * self.dt)
        #mean[4] = (ω * self.dt)
        #mean[5] = (dv * self.dt)
        jacobian = np.eye(self.ntotal) * self.dt
        #Add the following terms:
        #mean[2] = (da * self.dt)
        #mean[3] = (dh * self.dt)
        #mean[4] = (ω * self.dt)
        #mean[5] = (dv * self.dt)
        jacobian[2, 7] = self.dt
        jacobian[3, 8] = self.dt
        jacobian[4, 9] = self.dt
        jacobian[5, 6] = self.dt
        #Next, fill out these:
        #mean[0] = (v_w_ratio * (np.sin(θ + (ω * self.dt)) - np.sin(θ)))
        # Derivative of x w.r.t. θ is (v/ω) * (np.cos(θ + (self.dt * ω)) - np.cos(θ))
        jacobian[0, 4] = v_w_ratio * (np.cos(θ + (self.dt * ω)) - np.cos(θ))
        # Derivative of x w.r.t. v is (np.sin(θ + (ω * self.dt)) - np.sin(θ))/w
        jacobian[0, 5] = (np.sin(θ + (ω * self.dt)) - np.sin(θ)) / (ω + eps)
        # Derivative of x w.r.t. ω is -(v/ω^2) * (np.sin((self.dt * ω) + θ) - ((self.dt * w) * np.cos((self.dt * ω) + θ)) - np.sin(θ))
        jacobian[0, 9] = -(v/(((ω*ω) + eps))) * (np.sin((self.dt * ω) + θ) - ((self.dt * ω) * np.cos((self.dt * ω) + θ)) - np.sin(θ))
        #Finally. fill out these:
        #mean[1] = (v_w_ratio * (-np.cos(θ + (ω * self.dt)) + np.cos(θ)))
        # Derivative of y w.r.t. θ is (v_w_ratio * (np.sin(θ + (ω * self.dt)) - np.sin(θ)))
        jacobian[1, 4] = (v_w_ratio * (np.sin(θ + (ω * self.dt)) - np.sin(θ)))
        # Derivative of x w.r.t. v is (-np.cos(θ + (ω * self.dt)) + np.cos(θ))/w
        jacobian[1, 5] = (-np.cos(θ + (ω * self.dt)) + np.cos(θ))/(ω + eps)
        # Derivative of x w.r.t. ω is -(v/(ω*ω)) * (np.sin((self.dt * ω) + θ) - ((self.dt * ω) * np.cos((self.dt * ω) + θ)) - np.cos(θ))
        jacobian[1, 9] = -(v/((ω*ω) + eps)) * (np.sin((self.dt * ω) + θ) - ((self.dt * ω) * np.cos((self.dt * ω) + θ)) - np.cos(θ))




        std = [
            2 * self._std_weight_position * mean[3],
            2 * self._std_weight_position * mean[3],
            1e-2,
            2 * self._std_weight_position * mean[3],
            10 * self._std_weight_velocity * mean[3],
            10 * self._std_weight_velocity * mean[3],
            10 * self._std_weight_velocity * mean[3],
            1e-5,
            10 * self._std_weight_velocity * mean[3],
            10 * self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(std))        
        covariance = np.linalg.multi_dot((
            jacobian, covariance, jacobian.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (10 dimensional array).
        covariance : ndarray
            The state's covariance matrix (10x10 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx10 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx10x10 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        n = mean.shape[0]
        result_mean = np.zeros_like(mean)
        result_covariance = np.zeros_like(covariance)
        for i in range(n):
            result_mean[i], result_covariance[i] = self.predict(mean[i], covariance[i])
        return result_mean, result_covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')