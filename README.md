# Kalman-Filter
# Kalman filter in Python to predict the motion of an object moving in a plane with constant angular velocity
import numpy as np
import math
from decimal import Decimal
import matplotlib.pyplot as plt



# measurement_flag - a boolean indicating whether measurements are available

# This function returns two matrices:
# - Result_matrix: contains the navigation results over time
# - S_deviation_matrix: contains the corresponding standard deviations over time

def kalman_filter(measurement_flag):

#indices
	px_index = 0
	pz_index = 1
	vx_index = 2
	vz_index = 3
	angle_index = 4
	
	ax_index = 0
	az_index = 1
	
	
	T = 80
	time_step = 0.01
	
	#the state vector
	X_vector = np.array([100.00, 200.00, 20.00, 10.00, math.pi/9])
	
	#the covairance matrix of the dynamic model
	P_matrix = np.diag(np.array([10, 15, 2, 3, 0.08]) ** 2)
	
	#the covairance matrix of the measurement
	R_matrix = np.diag(np.array([8.0, 4.0]) ** 2)
	
	Result_vector = X_vector.copy()
	Result_vector = np.append(Result_vector, 0.0)
	Result_matrix = np.array([Result_vector])
	
	S_deviation_vector = np.sqrt(np.diag(P_matrix))
	S_deviation_vector = np.append(S_deviation_vector, 0.0)
	S_deviation_matrix = np.array([S_deviation_vector])
	

	wyb = 0.01
	
	#the current time of the navigation
	time = Decimal('0.01')
	
	#since Q is zero, we won't use any variable for it
	
	while(time <= 80):
	
		
		#calculating the current dynamic model
		
		fxb = 40*abs(math.sin(float(time)))
		fzb = 20*abs(math.cos(float(time)))
		theta = X_vector[angle_index]
		
		rot_matrix = np.array([
		[math.cos(theta), math.sin(theta)],
		[-math.sin(theta), math.cos(theta)]])
		
		b_a_vector = np.array([fxb, fzb])
		a_vector = rot_matrix @ b_a_vector


		X_vector[px_index] += (time_step*X_vector[vx_index] + 
			0.5*(time_step**2)*a_vector[ax_index])
					
		X_vector[pz_index] += (time_step*X_vector[vz_index] + 
			0.5*(time_step**2)*a_vector[az_index])

		X_vector[vx_index] += time_step*a_vector[ax_index] 
		X_vector[vz_index] += time_step*a_vector[az_index]
		X_vector[angle_index] += time_step*wyb
		
		der_rot_matrix = np.array([
			[-math.sin(theta), math.cos(theta)],
			[-math.cos(theta), -math.sin(theta)]])
		
		a_vector_der = der_rot_matrix @ b_a_vector
		
		F_matrix = np.array([
			[1, 0, time_step, 0, 0.5*(time_step**2)*a_vector_der[ax_index]],
			[0, 1, 0, time_step, 0.5*(time_step**2)*a_vector_der[az_index]],
			[0, 0, 1, 0, time_step*(a_vector_der[ax_index])],
			[0, 0, 0, 1, time_step*(a_vector_der[az_index])],
			[0, 0, 0, 0, 1]])
			
		P_matrix = F_matrix @ P_matrix @ F_matrix.T
		
		
		#measurement
		if(measurement_flag and time >= 10 and time%1 == 0):
			H_matrix = np.array([[1, 0, 0, 0, 0],[0, 1, 0, 0, 0]])	

			S_matrix = H_matrix @ P_matrix @ H_matrix.T + R_matrix
			
			#this vector represents the difference between the dynamic proccess
			#and the measurement
			diff_p_vector = np.array([
				np.random.normal(0, 8.0),
				np.random.normal(0, 4.0)])
			
			K_matrix = P_matrix @ H_matrix.T @ np.linalg.inv(S_matrix)
			X_vector += (K_matrix @ diff_p_vector)
			I = np.eye(5)
			
           #Joseph form
			P_matrix = (
				(I - K_matrix @ H_matrix)@ P_matrix@ (I - K_matrix @ H_matrix).T
    			+ K_matrix @ R_matrix @ K_matrix.T)
			
			
		Result_vector = X_vector.copy()
		Result_vector = np.append(Result_vector, time)
		Result_matrix = np.vstack([Result_matrix, Result_vector])
		
		S_deviation_vector = np.sqrt(np.diag(P_matrix))
		S_deviation_vector = np.append(S_deviation_vector, time)
		S_deviation_matrix = np.vstack([S_deviation_matrix, S_deviation_vector])
			

		time += Decimal('0.01')

	return Result_matrix, S_deviation_matrix
	
	
def kalman_diff_plot(Kalman_matirx, Refernce_matrix):
	Diff_matrix = Kalman_matirx[:, :-1] - Refernce_matrix[:, :-1]
	component_names = ['X_diff', 'Z_diff', 'Vx_diff', 'Vz_diff', 'Angle_diff']
	time = Kalman_matirx[:, 5] 


	for i in range(Diff_matrix.shape[1]):
		if i == 5:
			continue  
		plt.plot(time, Diff_matrix[:, i], '.', label= component_names[i])

	plt.xlabel('Time')
	plt.ylabel('Component differences')
	plt.title('Component differences as a function of time')
	plt.legend()
	plt.grid(True)
	plt.show()	

def s_deviation_plot(S_deviation__matrix):
	component_names = ['X', 'Z', 'Vx', 'Vz', 'Angle']
	time = S_deviation__matrix[:, 5] 


	for i in range(S_deviation__matrix.shape[1]):
		if i == 5:
			continue  
		plt.plot(time, S_deviation__matrix[:, i], '.', label=component_names[i])

	plt.xlabel('Time')
	plt.ylabel('Standard deviations')
	plt.title('Standard deviations as a function of time')
	plt.legend()
	plt.grid(True)
	plt.show()	
	
		
reference,p_r = kalman_filter(False)
kal, p_k =  kalman_filter( True)

kalman_diff_plot(kal, reference)
s_deviation_plot(p_k)


