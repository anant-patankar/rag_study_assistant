# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Anant Patankar

"""
RAG Study Assistant Demo Script

This script demonstrates the basic functionality of the RAG Study Assistant system.
It processes a document, indexes it, and allows users to query the system.
"""

import os
import argparse
import logging
from pathlib import Path

from src.config import RagSystemConfig, get_default_config, get_lightweight_config
from src.rag_system import RagSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('rag_demo')

def main():
    parser = argparse.ArgumentParser(description='RAG Study Assistant Demo')
    parser.add_argument('--document', '-d', type=str, help='Path to the document to process')
    parser.add_argument('--query', '-q', type=str, help='Query to ask about the document')
    parser.add_argument('--config', '-c', type=str, choices=['default', 'lightweight', 'custom'], 
                        default='default', help='Configuration to use')
    parser.add_argument('--custom_config', type=str, help='Path to custom configuration JSON file')
    parser.add_argument('--cache_dir', type=str, default='./rag_cache', 
                        help='Directory to store cache files')
    
    args = parser.parse_args()
    
    # Select configuration
    if args.config == 'lightweight':
        config = get_lightweight_config()
    elif args.config == 'custom' and args.custom_config:
        config = RagSystemConfig.load_from_file(args.custom_config)
    else:
        config = get_default_config()
    
    # Update cache directory
    config.cache_dir = args.cache_dir
    os.makedirs(config.cache_dir, exist_ok=True)
    
    # Initialize the RAG system
    logger.info("Initializing RAG system...")
    rag = RagSystem(config)
    
    # Process document if provided
    if args.document:
        doc_path = Path(args.document)
        if not doc_path.exists():
            logger.error(f"Document not found: {args.document}")
            return
        
        logger.info(f"Processing document: {doc_path}")
        rag.process_document(str(doc_path))
        logger.info("Document processing completed")
    
    # Run in interactive mode if no query is provided
    if not args.query:
        logger.info("Entering interactive query mode (type 'exit' to quit)")
        
        while True:
            query = input("\nYour question: ")
            if query.lower() in ['exit', 'quit']:
                break
                
            if not query.strip():
                continue
                
            try:
                result = rag.query(query)
                print("\nAnswer:")
                print(result.answer)
                
                print("\nSources:")
                for i, chunk in enumerate(result.context_chunks[:3]):
                    print(f"{i+1}. {chunk.source} - {chunk.section_title}")
                    
            except Exception as e:
                logger.error(f"Error processing query: {e}")
    
    # Process a single query if provided
    elif args.query:
        try:
            result = rag.query(args.query)
            print("\nAnswer:")
            print(result.answer)
            
            print("\nSources:")
            for i, chunk in enumerate(result.context_chunks[:3]):
                print(f"{i+1}. {chunk.source} - {chunk.section_title}")
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
    
    # Clean up
    rag.close()
    logger.info("RAG system closed")

if __name__ == "__main__":
    main()