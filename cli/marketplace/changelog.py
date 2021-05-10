from datetime import datetime


class ChangeLog:
    def __init__(self):
        self.changes = []
        self.changes_available = False

    def new_item(self, item_name: str, item_version: str):
        self.changes_available = True
        self.changes.append(
            f"New item created: `{item_name}` (version: `{item_version}`)"
        )

    def update_item(self, item_name: str, new_version: str, old_version: str):
        self.changes_available = True
        self.changes.append(
            f"Item Updated: `{item_name}` (from version: `{old_version}` to `{new_version}`)"
        )

    def compile(self) -> str:
        compiled = (
            f"### Change log [{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}]\n"
        )
        for i, change in enumerate(self.changes, start=1):
            compiled += f"{i}. {change}\n"
        compiled += "\n"
        return compiled
