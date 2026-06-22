#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib/core';
import { InfrastructureStack } from '../lib/infrastructure-stack';

const app = new cdk.App();
new InfrastructureStack(app, 'LlmGpuSizerStack', {
  env: { region: 'ap-southeast-7' },
});
