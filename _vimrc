nnoremap <C-c><C-c> :w<CR>:!conda activate tensorflow1_env && python %<CR>
nnoremap <F5> :w<CR>:!conda activate tensorflow1_env && python %<CR>

let g:ycm_python_interpreter_path = 'C:/Users/taku/miniconda3/envs/tensorflow1_env/python.exe'
let g:ycm_python_sys_path = []
let g:ycm_extra_conf_vim_data = [
  \  'g:ycm_python_interpreter_path',
  \  'g:ycm_python_sys_path'
  \]
let g:ycm_global_ycm_extra_conf = '~/global_extra_conf.py'
let g:ycm_always_populate_location_list = 1
