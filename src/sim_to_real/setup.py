from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'sim_to_real'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
         (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jacopo',
    maintainer_email='cionix90@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'traj_generator = sim_to_real.traj_generator:main',
            'nn_traj_generator = sim_to_real.traj_gen_nn:main'
        ],
    },
)
