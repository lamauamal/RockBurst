""""""
import os
import pandas as pd
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

load_dotenv()
MERGECSV = os.path.join(project_root, os.getenv("DATA"))

class RB_Criteria():
    def __init__(self, datapath):
        self.data = pd.read_csv(datapath)

    def criteria_SCF_1(self):
        """
        0: SCF<0.2
        1: 0.2<=SCF<0.3
        2: 0.3<=SCF<0.55
        3: SCF>=0.55
        """

    def criteria_SCF_2(self):
        """
        0: SCF<0.3
        1: 0.3<=SCF<0.5
        2: 0.5<=SCF<0.7
        3: SCF>=0.7
        """

    def criteria_Rc(self):
        """
        0: σc<80
        1: 80<=σc<120
        2: 120<=σc<180
        3: σc>=180
        """

    def criteria_B1(self):
        """
        0: B1<15
        1: 15<=B1<18
        2: 18<=B1<22
        3: B1>=22
        """

    def criteria_Wet(self):
        """
        0: Wet<2.0
        1: 2.0<=Wet<3.5
        2: 3.5<=Wet<5.0
        3: Wet>=5.0
        """

