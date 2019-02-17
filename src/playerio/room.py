import socket

from ._deserializer import Deserializer
from ._serializer import Serializer
from .event_handler import EventHandler
from .message import Message


class Room:

    def __init__(self, room_info):
        self.__room_id = room_info.id

        endpoint = [x for x in room_info.endpoints if x.port == 8184][0]

        self.__socket = socket.create_connection((endpoint.address, endpoint.port))
        self.__socket.settimeout(10)

        self.__socket.send('\0'.encode())
        self.send('join', room_info.join_key)

        self.__deserializer = Deserializer(self.__socket, self.__broadcast_message)

    @property
    def id(self):
        return self.__room_id

    @property
    def connected(self):
        return self.__deserializer.connected

    def disconnect(self):
        self.__deserializer.disconnect()

    def send(self, message, *args):
        m = message
        if type(message) != Message:
            m = Message(message, *args)
        print(f'Sending {m}')
        self.__socket.send(Serializer.serialize_message(m))

    def __broadcast_message(self, message):
        if message.type == 'playerio.joinresult':
            return
        EventHandler.broadcast(self, message)
