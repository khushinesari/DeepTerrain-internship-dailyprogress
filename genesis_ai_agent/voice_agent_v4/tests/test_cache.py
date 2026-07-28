from modules.mission_cache import mission_cache

mission_cache.refresh()

mission_cache.print_summary()

missions = mission_cache.get_all()

print(type(missions))
