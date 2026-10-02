project(
  'tandem-rng',
  'c', 'cython',
  version: '0.1.0',
  license: 'Apache-2.0',
  meson_version: '>=1.3',
  default_options: ['c_std=c99', 'buildtype=release', 'warning_level=2'],
)

py = import('python').find_installation(pure: false)
# NumPy 2 ships a pkg-config file. Without pkg-config, ask Python for the include directory.
numpy_dep = dependency('numpy', required: false)
if not numpy_dep.found()
  numpy_include = run_command(
    py, ['-c', 'import numpy; print(numpy.get_include())'], check: true,
  ).stdout().strip()
  numpy_dep = declare_dependency(include_directories: include_directories(numpy_include))
endif

py.install_sources(['src/tandem_rng/__init__.py'], subdir: 'tandem_rng')
py.extension_module(
  '_tandem',
  ['src/tandem_rng/_tandem.pyx', 'src/tandem_rng/c/tandem.c'],
  include_directories: ['src/tandem_rng/c'],
  dependencies: [numpy_dep],
  c_args: ['-DNPY_NO_DEPRECATED_API=NPY_2_0_API_VERSION'],
  install: true,
  subdir: 'tandem_rng',
)
