from mlrun import get_or_create_ctx
from kubernetes import config, client
from kubernetes.stream import stream

def main(context):
    context.logger.info("Starting to cmd")
    cli = K8SClient(context.logger)
    
    cmd = 'spark-submit'
    
    cli.exec_shell_cmd(cmd)

class K8SClient(object):

    def __init__(self, logger, namespace='default-tenant', config_file=None):
        self.namespace = namespace
        self.logger = logger
        self._init_k8s_config(config_file)
        self.v1api = client.CoreV1Api()

    def _init_k8s_config(self, config_file):
        try:
            config.load_incluster_config()
            self.logger.info('using in-cluster config.')
        except Exception:
            try:
                config.load_kube_config(config_file)
                self.logger.info('using local kubernetes config.')
            except Exception:
                raise RuntimeError(
                    'cannot find local kubernetes config file,'
                    ' place it in ~/.kube/config or specify it in '
                    'KUBECONFIG env var')

    def get_shell_pod_name(self):
        shell_pod = self.v1api.list_namespaced_pod(namespace=self.namespace)
        for i in shell_pod.items:
            if "shell" in i.metadata.name:
                self.logger.info("%s\t%s\t%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name))
                shell_name = i.metadata.name
                break
        return shell_name

    def exec_shell_cmd(self, cmd):
        shell_name = self.get_shell_pod_name()
        # Calling exec and waiting for response
        exec_command = [
            '/bin/bash',
            '-c',
            cmd]
        resp = stream(self.v1api.connect_get_namespaced_pod_exec,
                      shell_name,
                      self.namespace,
                      command=exec_command,
                      stderr=True, stdin=False,
                      stdout=True, tty=False)
        self.logger.info("Response: " + resp)


if __name__ == '__main__':
    context = get_or_create_ctx('spark-runner')
    main(context)
