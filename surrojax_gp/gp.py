
# import numbers
# from math import pi
# import numpy as np
# import jax.numpy as jnp
# from arraytainers import Jaxtainer

# class SingleOutputGP:

#     _no_operator_key = 'y'
#     _x1_key = 'x_1 only'
#     _x1_x2_key = 'x_1 and x_2'
#     _lb_key = 'lb'
#     _ub_key = 'ub'
#     _max_param = 1e1
#     _min_param = -1*_max_param

#     #
#     #   Constructor Methods
#     #

#     def __init__(self, kernel, params, data, mean=None, lb=None, ub=None, preprocess=None):
#         self._check_data(data)
#         self._data_dict = {self._no_operator_key: data}
#         # _kernel_dict = {'data': {}, '': {'first_arg', 'both_args':}}
#         self._mean_dict = {self._no_operator_key: mean}
#         self._kernel_dict = {self._no_operator_key: kernel}
#         self._params = Jaxtainer(params)
    
#     #
#     #   Kernel Methods
#     #

#     def kernel(self, x_1, x_2, params=None):
#         if params are None:
#             params = self._params
        

#     #
#     #   Function Construction Methods
#     #

#     def _update_kernel_dict(self, kernel, operator=None, operator_name=None):
        
#         self._check_operator_name(operator_name)
#         new_kernel_funcs = {}

#         if operator is not None:
#             kernel_x1, kernel_x1x2 = self._apply_operator(kernel, operator)
#             new_kernel_funcs[operator_name] = {self._x1_key: kernel_x1, self._x1_x2_key: kernel_x1x2}

#         for key, kernel in kernel_dict.items():
#             kernel_dict[key] = self._vectorise_kernel(kernel)

#         return kernel_dict

#     def _check_operator_name(self, operator_name):
#         if operator_name == self._data_key:
#             raise ValueError(f"Cannot name an operator '{self._data_key}'; please select another name")
#         if 

#     @staticmethod
#     def _swap_x1_and_x2(kernel):
#         return lambda x_2, x_1, params : kernel(x_1, x_2, params)

#     @staticmethod
#     def _apply_operator(kernel, operator):
#         kernel_x1 = operator(kernel)
#         kernel_x1x2 = swap_x1_and_x2(operator(swap_x1_and_x2(kernel_x1)))
#         return kernel_x1, kernel_x1x2

#     @staticmethod
#     def _vectorise_kernel(kernel):
#         return jax.vmap(jax.vmap(kernel, in_axes=(None,0,None)), in_axes=(0,None,None))

#     @staticmethod
#     def _differentiate_wrt_params(kernel):
#         return jax.jacfwd(kernel, argnums=2)

#     @staticmethod
#     def _create_kernel_matrix_func(kernel_dict):

#         def kernel_matrix(x_1, x_2, params):
#             pass
        
#         return kernel_matrix

#     #
#     #   Prediction Method
#     #

#     @staticmethod
#     def _get_combinations():


#     #
#     #   Negative Log-Likelihood Methods
#     #

#     def _get_x(self, x):
#         if x is None:
#             x = self.training_x
#         return x

#     def logprob(self, x=None):
#         x = self._get_x
#         y = self._construct_y()
#         return -0.5* - 0.5*self.data_len*jnp.log(2*pi)

#     def logprob_del_params(self):
#         pass

#     def neglogprob(self):
#         return -1*self.logprob()

#     def neglogprob_del_params(self):
#         return -1*self.logprob_del_params()

#     #
#     #   Data Methods
#     #

#     def _check_data(self, data):
#         if not isinstance(data, (np.array, jnp.array)):
#             raise TypeError('Data must be either a Numpy or a Jax Numpy array.')
#         if data.ndim != 2:
#             raise ValueError('Data must be a 2D array, where rows are observations and columns are features.')

#     def add_data(self, data, operator=None, operator_name=None):

#         self._check_data(data)

#     def training_x(self):
#         return 

#     @property
#     def num_operators(self):
#         return 

#     @property
#     def data(self):
#         return self._data
    
#     @property
#     def data_len(self):
#         return self.data.shape[0]

#     #
#     #   Python Operator Methods
#     #
    
#     def apply_operation(self, operator):
#         pass self.__class__(operator )

#     def __matmul__(self, operator):
#         return self.apply_operation(operator)

#     def __add__(self, val):
#         if isinstance(val, numbers.Number):


#     def __sub__(self, val):
#         return self.__add__(-1*val)


#     def __mult__(self):
#         pass

#     def __truediv__(self, val):
#         return self.__mult__(1/val)
        