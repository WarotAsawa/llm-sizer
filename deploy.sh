#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo -e "${PURPLE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║${NC}  🚀 ${CYAN}LLM GPU Sizer — Production Deploy${NC}       ${PURPLE}║${NC}"
echo -e "${PURPLE}║${NC}  📍 ${YELLOW}Region: ap-southeast-7${NC}                  ${PURPLE}║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════════════╝${NC}"
echo ""

# Get the S3 bucket and distribution ID from CloudFormation outputs
get_stack_resources() {
    BUCKET=$(aws cloudformation describe-stack-resources \
        --stack-name LlmGpuSizerStack \
        --region ap-southeast-7 \
        --query "StackResources[?ResourceType=='AWS::S3::Bucket'].PhysicalResourceId" \
        --output text 2>/dev/null)
    DIST_ID=$(aws cloudformation describe-stack-resources \
        --stack-name LlmGpuSizerStack \
        --region ap-southeast-7 \
        --query "StackResources[?ResourceType=='AWS::CloudFront::Distribution'].PhysicalResourceId" \
        --output text 2>/dev/null)
}

if [[ "$1" == "--quick" || "$1" == "-q" ]]; then
    echo -e "⚡ ${YELLOW}Quick deploy — frontend files only${NC}"
    echo ""

    get_stack_resources

    if [[ -z "$BUCKET" || -z "$DIST_ID" ]]; then
        echo -e "${RED}❌ Stack not found. Run full deploy first.${NC}"
        exit 1
    fi

    echo -e "📂 ${BLUE}Uploading files to S3...${NC}"
    aws s3 sync "$SCRIPT_DIR" "s3://$BUCKET" \
        --exclude "infrastructure/*" \
        --exclude "node_modules/*" \
        --exclude ".git/*" \
        --exclude "*.md" \
        --exclude "deploy.sh" \
        --region ap-southeast-7

    echo -e "🔄 ${BLUE}Invalidating CloudFront cache...${NC}"
    aws cloudfront create-invalidation \
        --distribution-id "$DIST_ID" \
        --paths "/*" \
        --query 'Invalidation.Id' \
        --output text

    echo ""
    echo -e "${GREEN}✅ Quick deploy complete!${NC} (files uploaded + cache invalidated)"
    echo -e "🌐 ${CYAN}https://$(aws cloudfront get-distribution --id "$DIST_ID" --query 'Distribution.DomainName' --output text)${NC}"
    echo ""
else
    echo -e "📦 ${BLUE}Full CDK deploy${NC}"
    echo ""

    cd "$SCRIPT_DIR/infrastructure"

    echo -e "📦 ${BLUE}Installing dependencies...${NC}"
    npm install --silent

    echo -e "🔨 ${BLUE}Synthesizing CDK stack...${NC}"
    npx cdk synth --quiet

    echo -e "🚀 ${YELLOW}Deploying to AWS...${NC}"
    npx cdk deploy --require-approval never

    echo ""
    echo -e "${GREEN}✅ Full deployment complete!${NC}"
    echo -e "🌐 ${CYAN}Your site is live at the CloudFront URL above.${NC}"
    echo ""
fi

echo -e "${PURPLE}Usage:${NC}"
echo -e "  ./deploy.sh          Full CDK deploy (infra + code)"
echo -e "  ./deploy.sh --quick  Frontend only (S3 sync + CF invalidation)"
echo ""
