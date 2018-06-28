import logging
import socketserver
import argparse
import pickle
from set_expansion_demo import set_expand


class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        print("handling expand request")
        res = ''
        self.data = str(self.request.recv(10240).strip(), 'utf-8')
        logger.debug('request data: ' + self.data)
        if self.data == 'get_vocab':
            print('getting vocabulary')
            res = se.get_vocab()
        else:
            data = [x.strip() for x in self.data.split(',')]
            print('expanding)')
            res = se.expand(data)
        print('compressing response')
        packet = pickle.dumps(res)
        print('response length= ' + str(len(packet)))
        # length = struct.pack('!I', len(packet))
        # packet = length + packet
        print('sending response')
        self.request.sendall(packet)
        print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='expand_server.py')
    parser.add_argument('model_path', metavar='model_path', type=str,
                        help='a path to the w2v model file')
    parser.add_argument('--host',type=str, default='localhost',
                        help='set port for the server')
    parser.add_argument('--port', type=int, default=1234,
                        help='set port for the server')

    args = parser.parse_args()

    port = args.port
    model_path = args.model_path
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    logger = logging.getLogger("set_expantion_demo")
    logger.debug("initialize model\n")
    se = set_expand.SetExpand(model_path)
    logger.debug("loading server\n")
    HOST, PORT = args.host, port
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)
    logger.debug("server loaded\n")
    server.serve_forever()

