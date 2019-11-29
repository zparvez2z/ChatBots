
from aiohttp import web
from interact import run
routes = web.RouteTableDef()

@routes.get('/')
async def hello(request):
    return web.Response(text="Hello, There .... input after / ie: https://pzchatbot.pagekite.me/your input")

@routes.get('/{input}')
async def variable_handler(request):
    return web.Response(
        text="replay>, {}".format(run(request.match_info['input'])))
app = web.Application()
app.add_routes(routes)
web.run_app(app,port=7070)