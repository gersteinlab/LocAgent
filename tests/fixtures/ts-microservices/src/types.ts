export interface User {
  id: string;
  email: string;
}

export interface Order {
  id: string;
  userId: string;
  total: number;
}

export type PaymentStatus = 'authorized' | 'declined';
