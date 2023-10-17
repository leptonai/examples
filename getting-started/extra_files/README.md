# Handling Extra Files

If your photon / deployment requires a few extra files that are not part of the
main python file, we provide a lightweighted way to add these files in your photon
by specifying the `extra_files` field in the Photon.

In this example, the main photon class is defined in `main.py`, and we want to include
two files: a `content.txt` file that can be read by the photon, and a `dependency.py`
file that we want to import as a submodule. The `extra_files` field is a list that
specifies these two files.

During deployment time, these files will be unarchived and then placed in the current
working directory of the photon. You can use `os.getcwd()` to get the current working
directory.

To run the example, simply do:

    lep photon run -n extra_files_example -m main.py

See the source files for more detailed explanation of the example.
