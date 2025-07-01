from setuptools import setup


# get version
with open('src/mitotrek/version.py') as f:
    exec(f.read())


setup(name='mitotrek',
	  version=__version__,
	  description='Analysis framework for mitochondrial mutations called from single-cell data',
	  url='https://github.com/vincent6liu/mitotrek',
	  author='Vincent Liu',
	  author_email='liuv@stanford.edu',
	  package_dir={'': 'src'},
	  packages=['mitotrek'],
	  install_requires=[
		  'numpy>=1.14.2',
		  'pandas>=0.22.0',
		  'scipy>=1.0.1',
		  'matplotlib>=2.2.2',
		  'seaborn>=0.8.1'
	  ],
	  )