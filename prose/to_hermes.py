import paramiko
import os
    
key=paramiko.RSAKey.from_private_key_file('/george_tests/astep_private_key.pem')

def connect_click():
    global ssh
    ssh = paramiko.client.SSHClient()
    ssh.load_system_host_keys()
    ssh.connect('192.168.8.12',port=22, username='astep', pkey=key)
    return

def disconnect_click():
    sftp.close()
    ssh.close()
    return

def to_hermes(phot):

    connect_click()

    sftp = ssh.open_sftp()
    
    p=Path(phot)
    
    for i in os.listdir(p.parent):
        if i.endswith()
    sftp.put(phot, os.path.join('from_dmc', p.name))
    sftp.put(os.path.join(p.parent, 'lightcurve.png'), os.path.join('from_dmc', 'lightcurve.png'))
    
    disconnect_click()