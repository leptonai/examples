import os
import sys

from leptonai import Photon


class Main(Photon):
    # extra_files defines files that will be included in the Photon package.
    extra_files = ["dependency.py", "content.txt"]

    def init(self):
        # If you want to use the extra_files field to store python files / modules that
        # you then import, you will need to add the current working directory to the
        # python path.
        #
        # Note that you should NOT use "__file__" to get the current working directory,
        # as the underlying cloudpickle class implicitly replaces the __file__ variable,
        # making local and remote environment inconsistent.
        sys.path.append(os.getcwd())

    @Photon.handler
    def get_content_txt(self) -> str:
        """
        A simple function to return the content of content.txt.
        """
        with open(os.path.join(os.getcwd(), "content.txt"), "r") as f:
            return f.read()

    @Photon.handler
    def get_dependency_content(self) -> str:
        """
        A simple function to return the content defined inside dependency.py.
        """
        # As long as you have added cwd in the system path, you can import it without
        # problem.
        import dependency

        return dependency.content()

    @Photon.handler
    def cwd(self) -> str:
        """
        A simple function to return the current working directory.
        """
        return os.getcwd()
