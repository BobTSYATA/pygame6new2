import os
import ast
from collections import defaultdict

ROOT_DIR = "."

class ClassInfo:
    def __init__(self, name, filename, is_real_class=True):
        self.name = name
        self.filename = filename
        self.dependencies = set()
        self.associations = set()
        self.is_real_class = is_real_class  # False for file modules without classes

def extract_classes_and_usages(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return {}

    result = {}
    module_basename = os.path.splitext(os.path.basename(filepath))[0]
    module_node = ClassInfo(module_basename, filepath, is_real_class=False)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_info = ClassInfo(node.name, filepath)
            for body_item in node.body:
                if isinstance(body_item, ast.FunctionDef):
                    for arg in body_item.args.args:
                        if arg.annotation and isinstance(arg.annotation, ast.Name):
                            class_info.dependencies.add(arg.annotation.id)
                    for subnode in ast.walk(body_item):
                        if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name):
                            class_info.dependencies.add(subnode.func.id)
                        elif isinstance(subnode, ast.Attribute) and isinstance(subnode.value, ast.Name):
                            class_info.dependencies.add(subnode.value.id)
                        elif isinstance(subnode, ast.Name):
                            class_info.dependencies.add(subnode.id)
                elif isinstance(body_item, ast.AnnAssign) and isinstance(body_item.annotation, ast.Name):
                    class_info.associations.add(body_item.annotation.id)
            result[class_info.name] = class_info

        elif isinstance(node, (ast.Call, ast.Attribute, ast.Name)):
            # Collect all external symbols from module-level code
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                module_node.dependencies.add(node.func.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                module_node.dependencies.add(node.value.id)
            elif isinstance(node, ast.Name):
                module_node.dependencies.add(node.id)

    # Only keep the module node if no class is defined
    if not any(isinstance(n, ast.ClassDef) for n in tree.body):
        result[module_node.name] = module_node

    return result

def find_all_classes(directory):
    all_classes = {}
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".py"):
                path = os.path.join(root, fname)
                classes = extract_classes_and_usages(path)
                all_classes.update(classes)
    return all_classes

def generate_plantuml(classes):
    known_class_names = set(classes.keys())
    lines = ["@startuml", "skinparam classAttributeIconSize 0", ""]

    for cls in classes.values():
        stereotype = " <<module>>" if not cls.is_real_class else ""
        lines.append(f"class {cls.name}{stereotype} {{}}")

    for cls in classes.values():
        for dep in cls.dependencies:
            if dep in known_class_names and dep != cls.name:
                # Flip the arrow: dependency -> class that uses it
                lines.append(f"{dep} ..> {cls.name}")
        for assoc in cls.associations:
            if assoc in known_class_names and assoc != cls.name:
                lines.append(f"{cls.name} --> {assoc}")

    lines.append("\n@enduml")
    return "\n".join(lines)

if __name__ == "__main__":
    classes = find_all_classes(ROOT_DIR)
    output = generate_plantuml(classes)

    with open("class_diagram.puml", "w", encoding="utf-8") as f:
        f.write(output)

    print(f"Found {len(classes)} classes/modules.")
    print("Diagram written to class_diagram.puml")

