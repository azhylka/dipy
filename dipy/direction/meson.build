cython_sources = [
  'bootstrap_direction_getter',
  'closest_peak_direction_getter',
  'probabilistic_direction_getter',
  'ptt_direction_getter',
  ]

cython_headers = [
  'closest_peak_direction_getter.pxd',
  'pmf.pxd',
  'probabilistic_direction_getter.pxd',
  ]

foreach ext: cython_sources
  if fs.exists(ext + '.pxd')
    extra_args += ['--depfile', meson.current_source_dir() +'/'+ ext + '.pxd', ]
  endif
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, inc_local],
    dependencies: [omp],
    install: true,
    subdir: 'dipy/direction'
  )
endforeach


# ── pmf: core PMF generators including INRPmfGen (libtorch TorchScript) ──────
# INRPmfGen is compiled into pmf.so.  When libtorch is available it is linked
# against the real inr_torch_helper.cpp; otherwise a no-op stub is used so
# that pmf.so still loads — INRPmfGen.__init__ will raise RuntimeError.
_torch_query = run_command(
  py3, '-c',
  'import os, torch; d=os.path.dirname(torch.__file__); print(d)',
  check: false,
)

if _torch_query.returncode() == 0
  _torch_dir = _torch_query.stdout().strip()

  # Match the CXX ABI that PyTorch itself was compiled with.
  _torch_abi = run_command(
    py3, '-c',
    'import torch; print(int(torch.compiled_with_cxx11_abi()))',
    check: false,
  ).stdout().strip()
  _abi_flag = '-D_GLIBCXX_USE_CXX11_ABI=' + (_torch_abi == '1' ? '1' : '0')

  _torch_inc = [
    include_directories(_torch_dir / 'include'),
    include_directories(_torch_dir / 'include' / 'torch' / 'csrc' / 'api' / 'include'),
  ]
  _torch_lib      = _torch_dir / 'lib'
  _torch_cpp_args = cython_cpp_args + [_abi_flag]
  _torch_link_args = [
    '-L' + _torch_lib,
    '-ltorch', '-ltorch_cpu', '-lc10',
    '-Wl,-rpath,' + _torch_lib,
  ]

  py3.extension_module('pmf',
    [
      cython_gen.process('pmf.pyx'),
      'inr_torch_helper.cpp',
    ],
    c_args: cython_c_args,
    cpp_args: _torch_cpp_args,
    link_args: _torch_link_args,
    include_directories: [incdir_numpy, inc_local] + _torch_inc,
    dependencies: [omp],
    override_options: ['cpp_std=c++17'],
    install: true,
    subdir: 'dipy/direction',
  )
  message('INRPmfGen: enabled (libtorch at ' + _torch_dir + ')')
else
  py3.extension_module('pmf',
    [
      cython_gen.process('pmf.pyx'),
      'inr_torch_helper_stub.c',
    ],
    c_args: cython_c_args,
    include_directories: [incdir_numpy, inc_local],
    dependencies: [omp],
    install: true,
    subdir: 'dipy/direction',
  )
  message('INRPmfGen: disabled (PyTorch not found — stub linked)')
endif


python_sources = ['__init__.py',
  'peaks.py',
  ]


py3.install_sources(
  python_sources + cython_headers,
  pure: false,
  subdir: 'dipy/direction'
)


subdir('tests')