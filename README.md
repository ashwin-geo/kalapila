# Kalapila

Code to help create an obsidian canvas that maps function calls. 

1. Copy the python file to be read into this repository.
2. In `MapPythonFunction.ipynb`, define `python_file_name` based on the file to be read.
3. Run the cells in `MapPythonFunction.ipynb`
4. Open the folder `TheVault` as a vault in obsidian to view the canvas file inside it.


The parameters in 
```python 
py2c.generate_canvas_for_python_file_coloured(python_file=python_file,
                                              canvas_file=canvas_file,
                                              max_func_call=5,
                                              dir=target_dir,
                                              update_layout=False,
                                              levels_iters=10)
```
can be tweaked to suit your requirements.

**NB:** Optimal placement of the blocks is yet to be implemented. Right now the blocks corresponding to functions and the connections corresponding to function calls are automatically created in the canvas, After which you can **manually move the blocks around** to focus on the relations you are interested in.