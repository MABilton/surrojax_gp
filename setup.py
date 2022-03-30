from setuptools import setup, find_packages

setup(
   name='surrojax_gp',
   version='0.0.0',
   author='Matthew Bilton',
   author_email='Matt.A.Bilton@gmail.com',
   python_requires='>=3.7.12',
   packages=find_packages(exclude=('tests', 'examples')),
   url="https://github.com/MABilton/gp_oed_surrogate",
   install_requires=[
      'jax>=0.2.19',
      'jaxlib>=0.1.69',
      'numpy>=1.19.5',
      'scipy>=1.4.1'
      ]
)