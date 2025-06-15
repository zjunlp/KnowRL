
DATA_PATH="./data/your_data_file.json"  # Your JSON/JSONL data file
DB_PATH="./knowledge_base.db"           # Output database path
BATCH_SIZE=10000                        # Processing batch size
MAX_DOCS=""                             # Max documents (empty = all)
MAX_TOKEN_LENGTH=450                    # Token length limit

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Building Knowledge Base..."

# Check data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo -e "${RED}Error: Data file not found: $DATA_PATH${NC}"
    echo -e "${YELLOW}Update DATA_PATH in this script to your data file${NC}"
    echo "Expected format: {\"title\": \"...\", \"text\": \"...\"}"
    exit 1
fi

# Build command
CMD="python build_knowledge_base.py --data_path \"$DATA_PATH\" --db_path \"$DB_PATH\" --batch_size $BATCH_SIZE --max_token_length $MAX_TOKEN_LENGTH"
if [ ! -z "$MAX_DOCS" ]; then
    CMD="$CMD --max_docs $MAX_DOCS"
fi

echo "Data: $DATA_PATH -> Database: $DB_PATH"
echo "Running: $CMD"
echo ""

# Execute
eval $CMD

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Success! Database saved at: $DB_PATH${NC}"
    if [ -f "$DB_PATH" ]; then
        DB_SIZE=$(du -h "$DB_PATH" | cut -f1)
        echo -e "${GREEN}Size: $DB_SIZE${NC}"
    fi
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi





