"""Intelligent code search assistant with multi-step exploration strategies.

This module demonstrates best practices for using the MCP tools together
to navigate and understand codebases effectively.
"""

from typing import Dict, List, Optional, Any
from openai import OpenAI
from sqlalchemy import create_engine

from .config import SearchConfig
from .searcher import CodeSearcher


class CodeSearchAssistant:
    """Intelligent assistant for code exploration and triage.

    Implements multi-step search strategies that combine semantic search
    with structural navigation to find relevant code efficiently.
    """

    def __init__(self, config: SearchConfig):
        """Initialize assistant with configuration.

        Args:
            config: SearchConfig instance with database and OpenAI settings
        """
        self.config = config
        self.engine = create_engine(config.database_url)
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.searcher = CodeSearcher(self.engine, config.table_name)

    def triage_issue(
        self,
        issue_description: str,
        feature_hint: Optional[str] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """Triage an issue using multi-step code exploration.

        Strategy:
        1. List available features to understand codebase structure
        2. Semantic search for relevant code
        3. Group results by feature and class
        4. For top matches, get surrounding context
        5. Search for related code (usages, dependencies)

        Args:
            issue_description: Natural language description of the issue
            feature_hint: Optional feature to focus on
            max_results: Maximum results to return

        Returns:
            Comprehensive analysis with relevant code locations
        """
        results = {
            "query": issue_description,
            "feature_hint": feature_hint,
            "steps": [],
        }

        # Step 1: List features to understand codebase
        features = self.searcher.list_features()
        results["steps"].append({
            "step": "list_features",
            "description": "Discovered available features in codebase",
            "feature_count": len(features),
            "features": features,
        })

        # Step 2: Semantic search for initial matches
        embedding_resp = self.openai_client.embeddings.create(
            model=self.config.embedding_model,
            input=[issue_description]
        )
        query_embedding = embedding_resp.data[0].embedding

        initial_results = self.searcher.search(
            query_embedding=query_embedding,
            top_k=min(max_results, 10),
            feature=feature_hint,
        )

        results["steps"].append({
            "step": "semantic_search",
            "description": "Initial semantic search for relevant code",
            "result_count": len(initial_results),
            "results": initial_results,
        })

        # Step 3: Group by feature and class to understand structure
        features_found = {}
        classes_found = {}

        for result in initial_results:
            feature = result["feature"]
            class_name = result.get("class_name")

            if feature not in features_found:
                features_found[feature] = []
            features_found[feature].append(result)

            if class_name:
                if class_name not in classes_found:
                    classes_found[class_name] = []
                classes_found[class_name].append(result)

        results["steps"].append({
            "step": "group_results",
            "description": "Grouped results by feature and class",
            "features_found": list(features_found.keys()),
            "classes_found": list(classes_found.keys()),
        })

        # Step 4: For top 3 results, get surrounding context
        context_results = []
        for result in initial_results[:3]:
            if result.get("path") and result.get("start_line"):
                context = self.searcher.get_surrounding_context(
                    file_path=result["path"],
                    line_number=result["start_line"],
                    context_lines=3,
                    feature=result["feature"]
                )
                context_results.append({
                    "target": result,
                    "surrounding_context": context,
                })

        results["steps"].append({
            "step": "get_context",
            "description": "Retrieved surrounding context for top matches",
            "context_count": len(context_results),
            "contexts": context_results,
        })

        # Step 5: For each unique class found, explore all methods
        class_explorations = []
        for class_name in list(classes_found.keys())[:3]:  # Top 3 classes
            methods = self.searcher.search_by_class(
                class_name=class_name,
                feature=feature_hint,
                limit=20
            )
            class_explorations.append({
                "class_name": class_name,
                "method_count": len(methods),
                "methods": methods,
            })

        results["steps"].append({
            "step": "explore_classes",
            "description": "Explored methods in relevant classes",
            "classes_explored": len(class_explorations),
            "class_details": class_explorations,
        })

        # Step 6: Search for related code (other references to these classes)
        related_code = []
        for class_name in list(classes_found.keys())[:2]:  # Top 2 classes
            references = self.searcher.search_related_code(
                class_or_method=class_name,
                top_k=5,
                feature=feature_hint
            )
            related_code.append({
                "class_name": class_name,
                "reference_count": len(references),
                "references": references,
            })

        results["steps"].append({
            "step": "find_references",
            "description": "Found code that references relevant classes",
            "related_code": related_code,
        })

        # Summary
        results["summary"] = {
            "total_features_in_codebase": len(features),
            "relevant_features": list(features_found.keys()),
            "relevant_classes": list(classes_found.keys()),
            "initial_matches": len(initial_results),
            "classes_explored": len(class_explorations),
            "recommendation": self._generate_recommendation(
                features_found, classes_found, issue_description
            ),
        }

        return results

    def explore_feature(
        self,
        feature_name: str,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Comprehensively explore a feature/module.

        Strategy:
        1. Get all code in the feature
        2. Group by class
        3. Identify entry points and key classes
        4. Map dependencies

        Args:
            feature_name: Name of the feature to explore
            limit: Maximum code chunks to retrieve

        Returns:
            Comprehensive feature analysis
        """
        results = {
            "feature": feature_name,
            "steps": [],
        }

        # Get all code in the feature
        all_code = self.searcher.search_by_feature(feature_name, limit)

        results["steps"].append({
            "step": "get_feature_code",
            "description": f"Retrieved all code in {feature_name} feature",
            "chunk_count": len(all_code),
        })

        # Group by file and class
        files = {}
        classes = {}

        for chunk in all_code:
            file_path = chunk["path"]
            class_name = chunk.get("class_name")

            if file_path not in files:
                files[file_path] = []
            files[file_path].append(chunk)

            if class_name:
                if class_name not in classes:
                    classes[class_name] = []
                classes[class_name].append(chunk)

        results["steps"].append({
            "step": "organize_structure",
            "description": "Organized code by files and classes",
            "file_count": len(files),
            "class_count": len(classes),
            "files": list(files.keys()),
            "classes": list(classes.keys()),
        })

        # Identify key classes (most methods)
        key_classes = sorted(
            classes.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]

        results["summary"] = {
            "feature": feature_name,
            "total_files": len(files),
            "total_classes": len(classes),
            "key_classes": [
                {"class_name": name, "method_count": len(methods)}
                for name, methods in key_classes
            ],
            "files": list(files.keys()),
        }

        return results

    def find_implementation(
        self,
        method_or_class: str,
        search_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Find implementation and usage of a method or class.

        Strategy:
        1. Search by class name to find definition
        2. Search related code to find usages
        3. Get full file context for implementation
        4. Identify dependencies

        Args:
            method_or_class: Name of method or class to find
            search_query: Optional semantic query for context

        Returns:
            Implementation details and usage examples
        """
        results = {
            "target": method_or_class,
            "steps": [],
        }

        # Try as class name first
        class_results = self.searcher.search_by_class(
            class_name=method_or_class,
            limit=50
        )

        if class_results:
            results["steps"].append({
                "step": "found_as_class",
                "description": f"Found {method_or_class} as a class definition",
                "method_count": len(class_results),
                "methods": class_results,
            })

            # Get full file for first occurrence
            if class_results[0].get("path"):
                file_chunks = self.searcher.get_file_chunks(
                    class_results[0]["path"]
                )
                results["steps"].append({
                    "step": "get_full_file",
                    "description": "Retrieved complete file contents",
                    "file_path": class_results[0]["path"],
                    "chunk_count": len(file_chunks),
                    "chunks": file_chunks,
                })

        # Search for related code (usages)
        usages = self.searcher.search_related_code(
            class_or_method=method_or_class,
            top_k=20
        )

        results["steps"].append({
            "step": "find_usages",
            "description": f"Found references to {method_or_class}",
            "usage_count": len(usages),
            "usages": usages,
        })

        # If semantic query provided, do semantic search too
        if search_query:
            embedding_resp = self.openai_client.embeddings.create(
                model=self.config.embedding_model,
                input=[search_query]
            )
            query_embedding = embedding_resp.data[0].embedding

            semantic_results = self.searcher.search(
                query_embedding=query_embedding,
                top_k=5
            )

            results["steps"].append({
                "step": "semantic_search",
                "description": "Semantic search for related functionality",
                "result_count": len(semantic_results),
                "results": semantic_results,
            })

        results["summary"] = {
            "target": method_or_class,
            "found_as_class": len(class_results) > 0,
            "method_count": len(class_results),
            "usage_count": len(usages),
        }

        return results

    def _generate_recommendation(
        self,
        features_found: Dict[str, List],
        classes_found: Dict[str, List],
        issue_description: str
    ) -> str:
        """Generate a human-readable recommendation."""
        if not features_found:
            return "No directly relevant code found. Try broader search terms or explore features manually."

        top_feature = max(features_found.items(), key=lambda x: len(x[1]))
        top_class = max(classes_found.items(), key=lambda x: len(x[1])) if classes_found else None

        recommendation = f"Primary focus: '{top_feature[0]}' feature with {len(top_feature[1])} relevant matches."

        if top_class:
            recommendation += f" Start investigation in the '{top_class[0]}' class which has {len(top_class[1])} relevant methods."

        return recommendation


def main():
    """Example usage of the code search assistant."""
    import json
    from .config import get_config

    config = get_config()
    assistant = CodeSearchAssistant(config)

    # Example 1: Triage an issue
    print("=== TRIAGE ISSUE ===")
    issue = "User's pulse age calculation seems incorrect after latest update"
    result = assistant.triage_issue(issue, feature_hint="pulse_age")
    print(json.dumps(result, indent=2, default=str))

    # Example 2: Explore a feature
    print("\n=== EXPLORE FEATURE ===")
    result = assistant.explore_feature("vitamin_d")
    print(json.dumps(result, indent=2, default=str))

    # Example 3: Find implementation
    print("\n=== FIND IMPLEMENTATION ===")
    result = assistant.find_implementation(
        "PulseAgeAnalysisService",
        search_query="calculate pulse age from heart rate"
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
