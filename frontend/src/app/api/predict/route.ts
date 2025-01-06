import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { priorityDate, visaCategory, country } = body;

    // TODO: Connect to your Python ML service
    // For now, returning mock data
    const prediction = {
      waitTime: Math.floor(Math.random() * 24) + 12,
      confidence: Math.floor(Math.random() * 20) + 80,
    };
    
    return NextResponse.json(prediction);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to process prediction' },
      { status: 500 }
    );
  }
}