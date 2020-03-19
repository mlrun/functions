import mlrun

def test_function(
    FUNCTION: str,
    db_path: str = 'http://mlrun-api:8080',
    hub_url: str = None,
    params = {}
):
    import mlrun, os
    mlrun.mlconf.dbpath = db_path
    mlrun.mlconf.hub_url = os.environ["MLRUN_HUB_URL"] if hub_url is None else hub_url

    urls = [f'../{FUNCTION}/function.yaml', f'db://{FUNCTION}:latest', f'hub://{FUNCTION}']

    tasks = []
    for url in urls:
        try:
            rfn = mlrun.import_function(url).apply(mlrun.mount_v3io())
            tsk = rfn.run(mlrun.NewTask(**params))
            tasks.append(tsk)
        except Exception as e:
            print(url, str(e))
            tasks.append(f'error {url}, {e}')

        print("++++++++++++++++++++++++++++++++++++++++++++++++")