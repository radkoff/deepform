# coding: utf-8

"""
    OPIF Service Data API

    No description provided (generated by Swagger Codegen https://github.com/swagger-api/swagger-codegen)  # noqa: E501

    OpenAPI spec version: 0.9.0
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six


class FolderHistory(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'entity_folder_id': 'str',
        'folder_name': 'str',
        'history_status': 'str',
        'folder_history_id': 'int',
        'entity_id': 'str',
        'source_service_code': 'str',
        'folder_path': 'str',
        'created_ts': 'datetime',
        'last_update_ts': 'datetime'
    }

    attribute_map = {
        'entity_folder_id': 'entity_folder_id',
        'folder_name': 'folder_name',
        'history_status': 'history_status',
        'folder_history_id': 'folder_history_id',
        'entity_id': 'entity_id',
        'source_service_code': 'source_service_code',
        'folder_path': 'folder_path',
        'created_ts': 'created_ts',
        'last_update_ts': 'last_update_ts'
    }

    def __init__(self, entity_folder_id=None, folder_name=None, history_status=None, folder_history_id=None, entity_id=None, source_service_code=None, folder_path=None, created_ts=None, last_update_ts=None):  # noqa: E501
        """FolderHistory - a model defined in Swagger"""  # noqa: E501
        self._entity_folder_id = None
        self._folder_name = None
        self._history_status = None
        self._folder_history_id = None
        self._entity_id = None
        self._source_service_code = None
        self._folder_path = None
        self._created_ts = None
        self._last_update_ts = None
        self.discriminator = None
        if entity_folder_id is not None:
            self.entity_folder_id = entity_folder_id
        if folder_name is not None:
            self.folder_name = folder_name
        if history_status is not None:
            self.history_status = history_status
        if folder_history_id is not None:
            self.folder_history_id = folder_history_id
        if entity_id is not None:
            self.entity_id = entity_id
        if source_service_code is not None:
            self.source_service_code = source_service_code
        if folder_path is not None:
            self.folder_path = folder_path
        if created_ts is not None:
            self.created_ts = created_ts
        if last_update_ts is not None:
            self.last_update_ts = last_update_ts

    @property
    def entity_folder_id(self):
        """Gets the entity_folder_id of this FolderHistory.  # noqa: E501


        :return: The entity_folder_id of this FolderHistory.  # noqa: E501
        :rtype: str
        """
        return self._entity_folder_id

    @entity_folder_id.setter
    def entity_folder_id(self, entity_folder_id):
        """Sets the entity_folder_id of this FolderHistory.


        :param entity_folder_id: The entity_folder_id of this FolderHistory.  # noqa: E501
        :type: str
        """

        self._entity_folder_id = entity_folder_id

    @property
    def folder_name(self):
        """Gets the folder_name of this FolderHistory.  # noqa: E501


        :return: The folder_name of this FolderHistory.  # noqa: E501
        :rtype: str
        """
        return self._folder_name

    @folder_name.setter
    def folder_name(self, folder_name):
        """Sets the folder_name of this FolderHistory.


        :param folder_name: The folder_name of this FolderHistory.  # noqa: E501
        :type: str
        """

        self._folder_name = folder_name

    @property
    def history_status(self):
        """Gets the history_status of this FolderHistory.  # noqa: E501


        :return: The history_status of this FolderHistory.  # noqa: E501
        :rtype: str
        """
        return self._history_status

    @history_status.setter
    def history_status(self, history_status):
        """Sets the history_status of this FolderHistory.


        :param history_status: The history_status of this FolderHistory.  # noqa: E501
        :type: str
        """

        self._history_status = history_status

    @property
    def folder_history_id(self):
        """Gets the folder_history_id of this FolderHistory.  # noqa: E501


        :return: The folder_history_id of this FolderHistory.  # noqa: E501
        :rtype: int
        """
        return self._folder_history_id

    @folder_history_id.setter
    def folder_history_id(self, folder_history_id):
        """Sets the folder_history_id of this FolderHistory.


        :param folder_history_id: The folder_history_id of this FolderHistory.  # noqa: E501
        :type: int
        """

        self._folder_history_id = folder_history_id

    @property
    def entity_id(self):
        """Gets the entity_id of this FolderHistory.  # noqa: E501


        :return: The entity_id of this FolderHistory.  # noqa: E501
        :rtype: str
        """
        return self._entity_id

    @entity_id.setter
    def entity_id(self, entity_id):
        """Sets the entity_id of this FolderHistory.


        :param entity_id: The entity_id of this FolderHistory.  # noqa: E501
        :type: str
        """

        self._entity_id = entity_id

    @property
    def source_service_code(self):
        """Gets the source_service_code of this FolderHistory.  # noqa: E501


        :return: The source_service_code of this FolderHistory.  # noqa: E501
        :rtype: str
        """
        return self._source_service_code

    @source_service_code.setter
    def source_service_code(self, source_service_code):
        """Sets the source_service_code of this FolderHistory.


        :param source_service_code: The source_service_code of this FolderHistory.  # noqa: E501
        :type: str
        """

        self._source_service_code = source_service_code

    @property
    def folder_path(self):
        """Gets the folder_path of this FolderHistory.  # noqa: E501


        :return: The folder_path of this FolderHistory.  # noqa: E501
        :rtype: str
        """
        return self._folder_path

    @folder_path.setter
    def folder_path(self, folder_path):
        """Sets the folder_path of this FolderHistory.


        :param folder_path: The folder_path of this FolderHistory.  # noqa: E501
        :type: str
        """

        self._folder_path = folder_path

    @property
    def created_ts(self):
        """Gets the created_ts of this FolderHistory.  # noqa: E501


        :return: The created_ts of this FolderHistory.  # noqa: E501
        :rtype: datetime
        """
        return self._created_ts

    @created_ts.setter
    def created_ts(self, created_ts):
        """Sets the created_ts of this FolderHistory.


        :param created_ts: The created_ts of this FolderHistory.  # noqa: E501
        :type: datetime
        """

        self._created_ts = created_ts

    @property
    def last_update_ts(self):
        """Gets the last_update_ts of this FolderHistory.  # noqa: E501


        :return: The last_update_ts of this FolderHistory.  # noqa: E501
        :rtype: datetime
        """
        return self._last_update_ts

    @last_update_ts.setter
    def last_update_ts(self, last_update_ts):
        """Sets the last_update_ts of this FolderHistory.


        :param last_update_ts: The last_update_ts of this FolderHistory.  # noqa: E501
        :type: datetime
        """

        self._last_update_ts = last_update_ts

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(FolderHistory, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, FolderHistory):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
