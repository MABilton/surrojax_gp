
# See: https://www.cse.wustl.edu/~garnett/cse515t/spring_2017/files/lecture_notes/11.pdf

class Operator:

    def __init__(self, , is_linear=False):
        if not is_linear:
            operator = _linearise_operator()
        self._operator = operator
        self._is_linear = is_linear

    def from_function(cls, function, is_linear=False):        
        def operator(input_func):
            def operator_on_func(*args, **kwargs):
                return op_func(input_func(*args, **kwargs))
            return operator_on_func
        return operator
    
class Gradient(Operator):

    def __init__(self, x_idx, use_jacrev=False):
        if use_jacrev:
            operator = lambda input_func: jax.jacrev(input_func)
        else:
            operator = lambda input_func: jax.jacfwd(input_func)
        super().__init__(operator)

def grad_operator(func):
    rearr_func = lambda x_1_d, x_1, x_2: func(jnp.array([x_1_d, x_1]), x_2, params)
    grad_func = 
    rerearr_func = lambda x_1, x_2: rearr_func(x_1[], x_1[], x_2, params)

    return rerearr_func