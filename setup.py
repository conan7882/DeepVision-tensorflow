from setuptools import setup

setup(name='tensorcv',
      version='0.1',
      description=' ',
      url='https://github.com/conan7882/DeepVision-tensorflow',
      author='Qian Ge',
      author_email='geqian1001@gmail.com',
      packages=['tensorcv', 'tensorcv.utils', 'tensorcv.algorithms', 'tensorcv.callbacks',
                'tensorcv.data', 'tensorcv.dataflow', 'tensorcv.dataflow.dataset',
                'tensorcv.models', 'tensorcv.predicts', 'tensorcv.train', 'tensorcv.tfdataflow'],
      zip_safe=False)
