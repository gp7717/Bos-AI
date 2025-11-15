#!/usr/bin/env python3
"""
Interactive query testing script.

Usage:
    python test_query.py "Your query here"
    python test_query.py  # Interactive mode
"""

import json
import sys
from datetime import datetime
from typing import Optional

from Agents.core.models import AgentRequest
from Agents.orchestrator.service import run


def print_response(response, verbose: bool = False):
    """Pretty print the response."""
    print("\n" + "=" * 80)
    print("RESPONSE SUMMARY")
    print("=" * 80)
    
    print(f"\n📝 Answer:")
    print(f"   {response.answer[:500]}{'...' if len(response.answer) > 500 else ''}")
    
    print(f"\n🤖 Agents Used:")
    agents_used = {}
    for result in response.agent_results:
        agent_name = result.agent.value
        status = result.status.value
        agents_used[agent_name] = status
        print(f"   - {agent_name}: {status}")
    
    print(f"\n📊 Graphs Generated: {len(response.all_graphs)}")
    if response.graphs:
        for i, graph in enumerate(response.graphs, 1):
            print(f"   Graph {i}: {graph.chart_type} ({len(graph.data.get('data', []))} data points)")
    elif response.graph:
        print(f"   Graph: {response.graph.chart_type}")
    
    print(f"\n📋 Tabular Data: {'Yes' if response.data else 'No'}")
    if response.data:
        print(f"   Columns: {len(response.data.columns)}")
        print(f"   Rows: {len(response.data.rows)}")
    
    if response.planner:
        print(f"\n🎯 Planner Decision:")
        print(f"   Rationale: {response.planner.rationale[:200]}...")
        print(f"   Chosen Agents: {', '.join(response.planner.chosen_agents)}")
        print(f"   Confidence: {response.planner.confidence:.2f}")
    
    if verbose:
        print(f"\n📜 Trace Events: {len(response.trace)}")
        for event in response.trace[-5:]:  # Last 5 events
            print(f"   - {event.event_type.value}: {event.message[:100]}")
    
    print("\n" + "=" * 80)


def test_query(query: str, verbose: bool = False):
    """Test a single query."""
    print(f"\n🔍 Testing Query: {query}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        request = AgentRequest(question=query)
        start_time = datetime.now()
        
        response = run(request)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"⏱️  Duration: {duration:.2f} seconds")
        
        print_response(response, verbose)
        
        # Validation checks
        print("\n✅ VALIDATION CHECKS:")
        checks = []
        
        # Check if API agent was used for net profit queries
        if "net profit" in query.lower():
            agents = [r.agent for r in response.agent_results]
            if "api_docs" in [a.value for a in agents]:
                checks.append("✅ API agent used for net profit (correct!)")
            else:
                checks.append("❌ API agent NOT used for net profit (should prefer API)")
        
        # Check for multiple graphs
        if "separate" in query.lower() or "multiple" in query.lower() or "also" in query.lower():
            if len(response.all_graphs) > 1:
                checks.append(f"✅ Multiple graphs generated ({len(response.all_graphs)})")
            else:
                checks.append(f"⚠️  Only {len(response.all_graphs)} graph(s) generated (expected multiple)")
        
        # Check for parallel execution (would need trace analysis)
        if len(response.agent_results) > 1:
            checks.append(f"✅ Multiple agents executed ({len(response.agent_results)})")
        
        for check in checks:
            print(f"   {check}")
        
        return response
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def interactive_mode():
    """Run in interactive mode."""
    print("\n" + "=" * 80)
    print("INTERACTIVE QUERY TESTER")
    print("=" * 80)
    print("\nEnter queries to test. Type 'exit' or 'quit' to stop.")
    print("Type 'verbose' to toggle verbose mode.")
    print("Type 'examples' to see example queries.\n")
    
    verbose = False
    
    while True:
        try:
            query = input("\n💬 Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'verbose':
                verbose = not verbose
                print(f"   Verbose mode: {'ON' if verbose else 'OFF'}")
                continue
            
            if query.lower() == 'examples':
                print("\n📚 Example Queries:")
                examples = [
                    "Get the net profit graph for the last 4 months",
                    "What's our performance this month?",
                    "Show me sales trends and revenue breakdown as separate charts",
                    "Get sales data and API documentation for the orders endpoint",
                    "What's our revenue, ROAS, and ad spend for this month?",
                ]
                for i, ex in enumerate(examples, 1):
                    print(f"   {i}. {ex}")
                continue
            
            test_query(query, verbose)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode
        query = " ".join(sys.argv[1:])
        verbose = "--verbose" in sys.argv or "-v" in sys.argv
        test_query(query, verbose)
    else:
        # Interactive mode
        interactive_mode()

