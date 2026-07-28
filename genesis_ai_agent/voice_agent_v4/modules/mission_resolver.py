"""
mission_resolver.py

Handles all GET operations.
"""
import re

# Enhanced Mission Resolver
class MissionResolver:
    def resolve(self, mission: dict, command: dict):
        field = command["field"]
        target = command.get("target","")
        # Remove array indices like [0], [1], etc.
        field = re.sub(r"\[\d+\]", "", field)
        if field=="mission_details":
            return {"success":True,"field":"mission_details","value":mission}

        if target == "" and field.startswith("assigned_assets"):
            return {"success":True,"field":"assigned_assets","value":mission.get("assigned_assets",[])}

        if target=="" and field in ("checkpoint","checkpoints"):
            return {"success":True,"field":"checkpoint","value":mission.get("checkpoint",[])}

        if target=="" and field=="objectives":
            return {"success":True,"field":"objectives","value":mission.get("objectives",[])}

        if target=="":
            if field not in mission:
                return {"success":False,"message":f"{field} not found."}
            return {"success":True,"field":field,"value":mission[field]}

        asset=self._find_asset(mission,target)
        if asset is None:
            return {"success":False,"message":f"Asset '{target}' not found."}
        if field not in asset:
            return {"success":False,"message":f"'{field}' not found in '{target}'."}
        return {"success":True,"field":field,"target":target,"value":asset[field]}

    def _find_asset(self,node,target):
        if not isinstance(target,str):
            return None
        target=target.lower()
        if isinstance(node,dict):
            for k in ("asset_name","name"):
                if isinstance(node.get(k),str) and node[k].lower()==target:
                    return node
            for v in node.values():
                r=self._find_asset(v,target)
                if r is not None:
                    return r
        elif isinstance(node,list):
            for item in node:
                r=self._find_asset(item,target)
                if r is not None:
                    return r
        return None

mission_resolver=MissionResolver()
