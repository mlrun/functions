# Copyright 2019 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from mlrun import code_to_function

mysql_url = 'mysql+pymysql://rfamro@mysql-rfam-public.ebi.ac.uk:4497/Rfam'
mysql_query = 'select rfam_acc,rfam_id,auto_wiki,description,author,seed_source FROM family'


def test_run_sql_to_file():
    fn = code_to_function(name='test_sql_to_file',
                          filename="sql_to_file.py",
                          handler="sql_to_file",
                          kind="job",
                          )
    fn.run(params={'sql_query': mysql_query,
                           'database_url': mysql_url}
           , local=True

           )

