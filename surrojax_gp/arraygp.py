
# class ArrayGP:

#     def __init__(self, crosskernel, *gp_list, **gp_dict):
        
#         num_outputs = len(gp_list) + len(gp_dict)
#         if num_outputs < 2:
#             raise ValueError('Must specify at least two Single Output Gaussian' 
#                             f'Processes to create a Multiple Output Gaussian Process.')
        
#         self._crosskernel = crosskernel
#         self._gp_dict = { **{f'gp_{idx}': gp for idx, gp in enumerate(gp_list)}, **gp_dict}
#         self._data = {name: gp.data for name, gp in self._gp_dict.items()}

#     @property
#     def num_outputs(self):
#         return len(self._gp_dict)