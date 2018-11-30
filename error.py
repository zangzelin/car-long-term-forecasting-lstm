import numpy as np 
point_by_point_predictions =  np.loadtxt('zzl.csv')
print( 'error is {}'.format( np.sum(point_by_point_predictions - y_test)/np.sum(y_test)  ) )

error = 0
for i in range(len(point_by_point_predictions)):
    error += np.abs(point_by_point_predictions[i]-y_test[i])/y_test[i]

error /= len( point_by_point_predictions )
print(error)