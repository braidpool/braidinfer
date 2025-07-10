#!/usr/bin/env python3
"""
Example demonstrating multiple agents with different system prompts
sharing common context using ChunkedLLM.

This shows how different AI personas can analyze the same data
efficiently by reusing context chunks while maintaining distinct
system prompts.
"""

import os
from typing import Dict, List
from braidinfer import ChunkedLLM, ChunkType


class Agent:
    """Represents an AI agent with a specific role and perspective."""
    
    def __init__(self, name: str, role: str, system_prompt: str, llm: ChunkedLLM):
        self.name = name
        self.role = role
        self.llm = llm
        
        # Register this agent's system prompt
        self.system_chunk_id = llm.register_chunk(
            system_prompt,
            ChunkType.SYSTEM_PROMPT,
            metadata={"agent": name, "role": role}
        )
        
        print(f"Agent '{name}' initialized (system chunk: {self.system_chunk_id[:8]}...)")
    
    def analyze(self, context_chunks: List[str], question: str) -> Dict[str, str]:
        """Analyze the given context and answer the question from this agent's perspective."""
        # Register the question as a query chunk
        query_id = self.llm.register_chunk(
            question,
            ChunkType.QUERY,
            metadata={"agent": self.name}
        )
        
        # Generate response
        output = self.llm.generate_from_chunks(
            system_chunk_id=self.system_chunk_id,
            context_chunk_ids=context_chunks,
            query_chunk_id=query_id,
            sampling_params={"temperature": 0.7, "max_tokens": 200}
        )
        
        return {
            "agent": self.name,
            "role": self.role,
            "response": output['text']
        }


def multi_agent_analysis():
    """Demonstrate multiple agents analyzing the same data."""
    print("=== Multi-Agent Analysis Demo ===\n")
    
    # Initialize ChunkedLLM
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = ChunkedLLM(
        model_path,
        max_chunks=1000,
        enable_deduplication=True,
        enforce_eager=True
    )
    
    # Create agents with different perspectives
    agents = [
        Agent(
            name="DataAnalyst",
            role="Data Analysis Expert",
            system_prompt="""You are a data analyst expert. Focus on patterns, trends, 
            and statistical insights. Provide quantitative analysis and highlight 
            numerical findings.""",
            llm=llm
        ),
        Agent(
            name="BusinessStrategist", 
            role="Business Strategy Consultant",
            system_prompt="""You are a business strategy consultant. Focus on business 
            implications, market opportunities, and strategic recommendations. Think 
            about competitive advantage and growth potential.""",
            llm=llm
        ),
        Agent(
            name="RiskAnalyst",
            role="Risk Assessment Specialist", 
            system_prompt="""You are a risk assessment specialist. Focus on identifying 
            potential risks, vulnerabilities, and mitigation strategies. Be cautious 
            and thorough in your analysis.""",
            llm=llm
        ),
        Agent(
            name="TechAdvisor",
            role="Technical Implementation Advisor",
            system_prompt="""You are a technical implementation advisor. Focus on technical 
            feasibility, implementation challenges, and technology requirements. Provide 
            practical technical recommendations.""",
            llm=llm
        )
    ]
    
    print(f"\nCreated {len(agents)} specialized agents\n")
    
    # Shared context - Company quarterly report data
    context_data = [
        """Q4 2023 Company Performance Report:
        - Revenue: $12.5M (up 23% YoY)
        - Active users: 2.1M (up 45% YoY)
        - Customer acquisition cost: $45 (down 15% from Q3)
        - Churn rate: 5.2% (down from 6.8% in Q3)
        - New feature adoption: 68% of users
        - Server uptime: 99.2% (below 99.9% target)""",
        
        """Market Context:
        - Industry growing at 15% annually
        - Main competitor launched similar feature last month
        - New regulations coming in Q2 2024
        - Customer satisfaction: 4.2/5 (industry avg: 3.8/5)
        - Mobile usage: 73% of total traffic"""
    ]
    
    # Register context chunks (shared across all agents)
    context_chunk_ids = []
    for i, context in enumerate(context_data):
        chunk_id = llm.register_chunk(
            context,
            ChunkType.CONTEXT,
            metadata={"document": f"report_section_{i+1}"}
        )
        context_chunk_ids.append(chunk_id)
    
    print(f"Registered {len(context_chunk_ids)} shared context chunks\n")
    
    # Get initial stats
    initial_stats = llm.get_chunk_stats()
    
    # Have each agent analyze the data
    question = "What are the top 3 priorities for next quarter based on this data?"
    
    print(f"Question: {question}\n")
    print("=" * 70)
    
    analyses = []
    for agent in agents:
        print(f"\n{agent.name} ({agent.role}):")
        print("-" * 50)
        
        result = agent.analyze(context_chunk_ids, question)
        analyses.append(result)
        
        print(result['response'][:400] + "..." if len(result['response']) > 400 else result['response'])
    
    # Show efficiency gains
    final_stats = llm.get_chunk_stats()
    
    print("\n" + "=" * 70)
    print("\n=== Efficiency Analysis ===")
    print(f"Total agents: {len(agents)}")
    print(f"Shared context chunks: {len(context_chunk_ids)}")
    print(f"Total unique chunks: {final_stats['total_chunks']}")
    print(f"Cache hits: {final_stats['cache_hits']}")
    print(f"Hit rate: {final_stats['hit_rate']:.1%}")
    
    # Calculate memory savings
    chunks_without_sharing = len(agents) * (1 + len(context_chunk_ids) + 1)  # system + contexts + query per agent
    chunks_with_sharing = final_stats['total_chunks']
    saved = chunks_without_sharing - chunks_with_sharing
    
    print(f"\nMemory efficiency:")
    print(f"• Without context sharing: {chunks_without_sharing} chunks")
    print(f"• With context sharing: {chunks_with_sharing} chunks")
    print(f"• Reduction: {saved} chunks ({saved/chunks_without_sharing*100:.0f}%)")


def collaborative_analysis():
    """Demonstrate agents building on each other's analysis."""
    print("\n\n=== Collaborative Agent Analysis ===\n")
    
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = ChunkedLLM(
        model_path,
        max_chunks=1000,
        enable_deduplication=True,
        enforce_eager=True
    )
    
    # Create a research team
    researcher = Agent(
        name="Researcher",
        role="Research Analyst",
        system_prompt="You are a research analyst. Gather facts and identify key insights from data.",
        llm=llm
    )
    
    synthesizer = Agent(
        name="Synthesizer",
        role="Synthesis Expert",
        system_prompt="You are a synthesis expert. Combine multiple perspectives into coherent conclusions.",
        llm=llm
    )
    
    # Initial data
    data_chunk_id = llm.register_chunk(
        """Recent AI Development Trends:
        - 70% increase in enterprise AI adoption
        - Open source models gaining market share
        - Concerns about AI safety growing
        - Regulatory frameworks being developed globally
        - Skills gap widening in AI engineering""",
        ChunkType.CONTEXT
    )
    
    # Phase 1: Initial research
    print("Phase 1: Initial Research")
    research_output = researcher.analyze(
        [data_chunk_id],
        "What are the key opportunities and challenges in the AI market?"
    )
    print(f"Researcher: {research_output['response'][:300]}...")
    
    # Register researcher's output as new context
    research_chunk_id = llm.register_chunk(
        research_output['response'],
        ChunkType.CONTEXT,
        metadata={"source": "researcher_analysis"}
    )
    
    # Phase 2: Synthesis
    print("\n\nPhase 2: Synthesis")
    synthesis_output = synthesizer.analyze(
        [data_chunk_id, research_chunk_id],  # Include both original data and research
        "Based on the research analysis, what strategic recommendations would you make?"
    )
    print(f"Synthesizer: {synthesis_output['response'][:300]}...")
    
    # Show collaboration efficiency
    stats = llm.get_chunk_stats()
    print(f"\n\nCollaboration Statistics:")
    print(f"• Chunks created: {stats['total_chunks']}")
    print(f"• Chunks reused: {stats['cache_hits']}")
    print(f"• Knowledge building: Research → Synthesis")


def main():
    """Run multi-agent demonstrations."""
    # Check if model exists
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    if not os.path.exists(model_path):
        print("Error: Model not found!")
        print("Please download the model first:")
        print("huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/")
        return
    
    # Run demonstrations
    multi_agent_analysis()
    collaborative_analysis()
    
    print("\n\n🎉 Multi-agent demonstration completed!")
    print("\nKey takeaways:")
    print("1. Multiple agents can share the same context chunks efficiently")
    print("2. Each agent maintains its unique perspective via system prompts")
    print("3. Agents can build on each other's outputs")
    print("4. Significant memory savings when analyzing shared data")


if __name__ == "__main__":
    main()