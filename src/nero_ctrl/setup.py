import os
from glob import glob

from setuptools import setup


package_name = 'nero_ctrl'

setup(
	name=package_name,
	version='0.0.1',
	packages=[package_name],
	data_files=[
		(
			'share/ament_index/resource_index/packages',
			['resource/' + package_name],
		),
		(
			'share/' + package_name,
			['package.xml'],
		),
		(
			os.path.join('share', package_name, 'launch'),
			glob('launch/*.launch.py'),
		),
		(
			os.path.join('share', package_name, 'config'),
			glob('config/*.yaml'),
		),
	],
	install_requires=['setuptools', 'python-can'],
	zip_safe=True,
	entry_points={
		'console_scripts': [
			'can_module = nero_ctrl.can_module:main',
            'main_node = nero_ctrl.main_node:main',
            'controller_node = nero_ctrl.controller_node:main',
			'ik_node = nero_ctrl.IK_node:main',
			'calibration_node = nero_ctrl.calibration_module:main',
			'cube_tracker_node = nero_ctrl.cube_tracker_node:main',
			'intercept_node = nero_ctrl.intercept_node:main',
			'ultrasonic_sensor_node = nero_ctrl.ultrasonic_sensor_node:main',
			'velocity_estimate_node = nero_ctrl.velocity_estimate_node:main',
		],
	},
)
