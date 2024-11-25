from setuptools import setup 

setup( 
	name='TN4QA', 
	version='0.1', 
	description='A Python package to integrate tensor network methods with quantum algorithms.', 
	packages=['tn4qa'], 
    python_requires='<3.12.0',
	install_requires=[ 
		'numpy', 
		'scipy',
        'sparse', 
        'qiskit',
        'qiskit_ibm_provider',
        'qiskit_ibm_runtime',
        'qiskit-aer',
        'symmer',
        'cotengra',
        'kahypar',
        'block2'
	], 
) 
