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

