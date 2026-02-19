export enum EventType {
  UserCreated = 'user.created',
  OrderPlaced = 'order.placed',
  PaymentAuthorized = 'payment.authorized',
}

export interface DomainEvent {
  type: EventType;
  payload: Record<string, unknown>;
}

export function publish(event: DomainEvent): void {
  console.log('publish', event.type);
}
