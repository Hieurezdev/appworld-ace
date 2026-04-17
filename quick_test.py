import traceback
import sys

def mock_execution():
    def _api_request(**kwargs):
        raise TypeError("'NoneType' object is not iterable")

    class Spotify:
        login = _api_request
    
    class Apis:
        spotify = Spotify()
    apis = Apis()

    spotify_password = None
    # this line matches python-input
    login_result = apis.spotify.login(username='paul_mill@gmail.com', password=spotify_password)

try:
    mock_execution()
except Exception as e:
    import appworld.common.utils as utils
    # fake an ipython input frame
    e.__traceback__.tb_frame.f_code.co_filename = "<ipython-input-1-123>"
    print(utils.get_stack_trace_from_exception(e, only_ipython=True))
